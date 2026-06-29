from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import create_engine, text


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def resolve_chunk_ids(
    database_url: str,
    repo_id: str,
    input_path: Path,
    output_path: Path,
) -> None:
    engine = create_engine(database_url)

    rows = load_rows(input_path)
    resolved_rows = []

    with engine.connect() as connection:
        for row in rows:
            file_paths = [str(v) for v in row.get("relevant_file_paths", [])]
            symbols = [str(v) for v in row.get("relevant_symbols", [])]

            clauses = []
            params: dict[str, object] = {"repo_id": repo_id}

            if file_paths:
                file_placeholders = []
                for index, file_path in enumerate(file_paths):
                    key = f"file_path_{index}"
                    params[key] = file_path
                    file_placeholders.append(f":{key}")
                clauses.append(f"file_path IN ({', '.join(file_placeholders)})")

            if symbols:
                symbol_placeholders = []
                for index, symbol in enumerate(symbols):
                    key = f"symbol_{index}"
                    params[key] = symbol
                    symbol_placeholders.append(f":{key}")
                clauses.append(f"symbol_name IN ({', '.join(symbol_placeholders)})")

            if not clauses:
                row["relevant_chunk_ids"] = []
                resolved_rows.append(row)
                continue

            # A chunk is relevant when it matches one of the curated paths or symbols.
            sql = text(
                f"""
                SELECT chunk_id
                FROM chunks
                WHERE repo_id = :repo_id
                  AND ({' OR '.join(clauses)})
                ORDER BY file_path, start_line
                """
            )

            chunk_ids = [
                str(result.chunk_id)
                for result in connection.execute(sql, params)
            ]

            row["repo_id"] = repo_id
            row["relevant_chunk_ids"] = sorted(set(chunk_ids))
            resolved_rows.append(row)

    with output_path.open("w", encoding="utf-8") as f:
        for row in resolved_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(resolved_rows)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve curated Requests file paths/symbols into chunk IDs from rag.db."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--database-url",
        default="sqlite:///./data/rag.db",
    )
    parser.add_argument(
        "--input",
        default="data/eval/requests_eval_template.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/eval/requests_eval.jsonl",
    )
    args = parser.parse_args()

    resolve_chunk_ids(
        database_url=args.database_url,
        repo_id=args.repo_id,
        input_path=Path(args.input),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
