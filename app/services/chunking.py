from __future__ import annotations
import ast
import fnmatch
import re
from pathlib import Path
from typing import Iterable
from app.domain.models import CodeChunk
from app.utils.hashing import sha256_text

EXT_TO_LANGUAGE = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cs": "c_sharp",
    ".kt": "kotlin", ".swift": "swift", ".scala": "scala", ".sh": "bash",
    ".md": "markdown", ".rst": "rst", ".txt": "text", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".toml": "toml", ".sql": "sql",
}

TS_NODE_TYPES = {
    "function_definition", "function_declaration", "method_definition", "method_declaration",
    "class_definition", "class_declaration", "interface_declaration", "struct_item", "impl_item",
    "function_item", "method", "constructor_declaration", "module", "namespace_definition",
}


class ASTChunker:
    def iter_files(self, root: Path, include_globs: list[str], exclude_globs: list[str]) -> Iterable[Path]:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXT_TO_LANGUAGE:
                continue
            rel = path.relative_to(root).as_posix()
            if include_globs and not any(fnmatch.fnmatch(rel, p) for p in include_globs):
                continue
            if any(fnmatch.fnmatch(rel, p) for p in exclude_globs):
                continue
            if path.stat().st_size > 2_000_000:
                continue
            yield path

    def chunk_repository(self, repo_id: str, root: Path, include_globs: list[str], exclude_globs: list[str]) -> tuple[list[CodeChunk], int]:
        chunks: list[CodeChunk] = []
        files_seen = 0
        for path in self.iter_files(root, include_globs, exclude_globs):
            files_seen += 1
            text = path.read_text(encoding="utf-8", errors="ignore")
            rel = path.relative_to(root).as_posix()
            language = EXT_TO_LANGUAGE[path.suffix.lower()]
            file_chunks = self._chunk_python(repo_id, rel, text) if language == "python" else self._chunk_tree_sitter(repo_id, rel, text, language)
            if not file_chunks:
                file_chunks = self._fallback_chunks(repo_id, rel, text, language)
            chunks.extend(file_chunks)
        return chunks, files_seen

    def _chunk_python(self, repo_id: str, file_path: str, text: str) -> list[CodeChunk]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        lines = text.splitlines()
        imports = self._python_imports(tree)
        out: list[CodeChunk] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            content = "\n".join(lines[start - 1:end])
            deps = sorted(set(imports + self._python_symbol_dependencies(node)))
            out.append(self._make_chunk(repo_id, file_path, "python", getattr(node, "name", "<anonymous>"), type(node).__name__, start, end, content, deps))
        # Include module-level code/imports as a separate chunk for dependency context.
        if text.strip():
            module_end = min(len(lines), 160)
            module_content = "\n".join(lines[:module_end])
            out.append(self._make_chunk(repo_id, file_path, "python", file_path, "module", 1, module_end, module_content, imports))
        return self._dedupe(out)

    @staticmethod
    def _python_imports(tree: ast.AST) -> list[str]:
        deps: list[str] = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                deps.extend(alias.name for alias in n.names)
            elif isinstance(n, ast.ImportFrom):
                prefix = n.module or ""
                deps.extend(f"{prefix}.{alias.name}".strip(".") for alias in n.names)
        return deps

    @staticmethod
    def _python_symbol_dependencies(node: ast.AST) -> list[str]:
        deps: list[str] = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    deps.append(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    deps.append(n.func.attr)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                deps.append(n.id)
        return deps

    def _chunk_tree_sitter(self, repo_id: str, file_path: str, text: str, language: str) -> list[CodeChunk]:
        try:
            from tree_sitter_language_pack import get_parser
            parser = get_parser(language)
            tree = parser.parse(text.encode("utf-8"))
        except Exception:
            return []
        lines = text.splitlines()
        out: list[CodeChunk] = []

        def walk(node) -> None:
            if node.type in TS_NODE_TYPES:
                start, end = node.start_point[0] + 1, node.end_point[0] + 1
                content = "\n".join(lines[start - 1:end])
                name = self._ts_name(node, text) or f"{node.type}@{start}"
                deps = self._generic_dependencies(content)
                out.append(self._make_chunk(repo_id, file_path, language, name, node.type, start, end, content, deps))
                return
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        if text.strip():
            module_end = min(len(lines), 160)
            out.append(self._make_chunk(repo_id, file_path, language, file_path, "module", 1, module_end, "\n".join(lines[:module_end]), self._generic_dependencies(text[:12000])))
        return self._dedupe(out)

    @staticmethod
    def _ts_name(node, text: str) -> str | None:
        for field in ("name", "declarator"):
            child = node.child_by_field_name(field)
            if child is not None:
                return text.encode("utf-8")[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")[:200]
        return None

    def _fallback_chunks(self, repo_id: str, file_path: str, text: str, language: str) -> list[CodeChunk]:
        lines = text.splitlines()
        if not lines:
            return []
        window, overlap = 120, 20
        out = []
        for start_idx in range(0, len(lines), window - overlap):
            end_idx = min(len(lines), start_idx + window)
            content = "\n".join(lines[start_idx:end_idx])
            out.append(self._make_chunk(repo_id, file_path, language, f"{file_path}:{start_idx+1}", "fallback", start_idx + 1, end_idx, content, self._generic_dependencies(content)))
            if end_idx == len(lines):
                break
        return out

    @staticmethod
    def _generic_dependencies(content: str) -> list[str]:
        names = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", content)
        stop = {"class", "function", "return", "const", "public", "private", "static", "import", "from", "this", "self", "true", "false", "null", "none"}
        return sorted({n for n in names if n.lower() not in stop})[:100]

    @staticmethod
    def _make_chunk(repo_id: str, file_path: str, language: str, symbol_name: str, symbol_type: str, start: int, end: int, content: str, deps: list[str]) -> CodeChunk:
        content_hash = sha256_text(content)
        chunk_id = sha256_text(f"{repo_id}:{file_path}:{start}:{end}:{content_hash}")[:24]
        return CodeChunk(chunk_id, repo_id, file_path, language, symbol_name, symbol_type, start, end, content, content_hash, deps)

    @staticmethod
    def _dedupe(chunks: list[CodeChunk]) -> list[CodeChunk]:
        seen, out = set(), []
        for c in chunks:
            key = (c.file_path, c.start_line, c.end_line, c.content_hash)
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out
