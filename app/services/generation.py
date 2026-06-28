from __future__ import annotations
from openai import OpenAI
from app.core.config import get_settings
from app.domain.models import RetrievedChunk
from app.domain.schemas import Citation
from app.utils.tokens import count_tokens


class AnswerGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key) if self.settings.openai_api_key else None

    def build_context(self, results: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
        selected: list[RetrievedChunk] = []
        blocks: list[str] = []
        used = 0
        for i, r in enumerate(results, start=1):
            c = r.chunk
            block = f"[SOURCE {i}] file={c.file_path} lines={c.start_line}-{c.end_line} chunk_id={c.chunk_id} symbol={c.symbol_name}\n{c.content}"
            tokens = count_tokens(block)
            if selected and used + tokens > self.settings.max_context_tokens:
                break
            selected.append(r)
            blocks.append(block)
            used += tokens
        return "\n\n".join(blocks), selected

    def generate(self, question: str, results: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
        context, selected = self.build_context(results)
        citations = [Citation(file_path=r.chunk.file_path, start_line=r.chunk.start_line, end_line=r.chunk.end_line, chunk_id=r.chunk.chunk_id, symbol_name=r.chunk.symbol_name) for r in selected]
        if not selected:
            return "No indexed context matched the question.", []
        if self.client is None:
            preview = "\n\n".join(f"- {r.chunk.file_path}:{r.chunk.start_line}-{r.chunk.end_line} — {r.chunk.symbol_name}" for r in selected[:5])
            return f"OPENAI_API_KEY is not configured. The strongest grounded source matches are:\n{preview}", citations

        response = self.client.responses.create(
            model=self.settings.openai_model,
            input=[
                {"role": "system", "content": "You are an expert to answer questions about a codebase. Use only supplied sources. Be precise. Cite claims inline as [SOURCE n]. If evidence is insufficient, say so. Never invent files, symbols, or behavior."},
                {"role": "user", "content": f"Question:\n{question}\n\nSources:\n{context}"},
            ],
            temperature=0,
        )
        return response.output_text, citations
