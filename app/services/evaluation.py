from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from app.core.config import get_settings
from app.domain.schemas import EvalItem, EvalResponse, DeepEvalMetricResult
from app.services.retrieval import HybridRetriever


class EvaluationService:
    """
    Evaluates both retrieval and generation quality.

    Deterministic metrics:
        - Chunk-ID Precision@K
        - Chunk-ID Recall@K

    DeepEval metrics:
        - Contextual Precision
        - Contextual Recall
        - Faithfulness
        - Answer Relevancy
    """
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Any,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.settings = get_settings()

    # def evaluate(self, dataset_path: str, repo_id: str | None, k: int) -> EvalResponse:
    #     path = Path(dataset_path)
    #     if not path.exists():
    #         raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    #     rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    #     items: list[EvalItem] = []
    #     for row in rows:
    #         relevant = set(row.get("relevant_chunk_ids", []))
    #         retrieved = self.retriever.retrieve(row["question"], repo_id=repo_id or row.get("repo_id"), top_k=k)
    #         ids = [r.chunk.chunk_id for r in retrieved]
    #         hits = len(set(ids) & relevant)
    #         precision = hits / len(ids) if ids else 0.0
    #         recall = hits / len(relevant) if relevant else 0.0
    #         items.append(EvalItem(question=row["question"], precision_at_k=precision, recall_at_k=recall, retrieved_chunk_ids=ids, relevant_chunk_ids=sorted(relevant)))
    #     n = len(items) or 1
    #     return EvalResponse(
    #         mean_context_precision=sum(i.precision_at_k for i in items) / n,
    #         mean_context_recall=sum(i.recall_at_k for i in items) / n,
    #         items=items,
    #     )
    
    
    def evaluate(
        self,
        dataset_path: str,
        repo_id: str | None,
        k: int,
    ) -> EvalResponse:
        rows = self._load_dataset(dataset_path)

        items: list[EvalItem] = []

        for row in rows:
            item = self._evaluate_row(
                row=row,
                repo_id=repo_id,
                k=k,
            )
            items.append(item)

        return self._build_response(items)

    def _evaluate_row(
        self,
        row: dict[str, Any],
        repo_id: str | None,
        k: int,
    ) -> EvalItem:
        question = self._required_string(row, "question")

        resolved_repo_id = repo_id or row.get("repo_id")

        relevant_chunk_ids = {
            str(chunk_id)
            for chunk_id in row.get("relevant_chunk_ids", [])
        }

        expected_output = self._optional_string(
            row.get("expected_output")
        )

        retrieved = self.retriever.retrieve(
            question,
            repo_id=resolved_repo_id,
            top_k=k,
        )

        retrieved_chunk_ids = [
            result.chunk.chunk_id
            for result in retrieved
        ]

        retrieval_context = [
            self._format_retrieval_context(result)
            for result in retrieved
        ]

        actual_output = self._generate_answer(
            question=question,
            retrieved=retrieved,
        )

        exact_hits = len(
            set(retrieved_chunk_ids) & relevant_chunk_ids
        )

        precision_at_k = (
            exact_hits / len(retrieved_chunk_ids)
            if retrieved_chunk_ids
            else 0.0
        )

        recall_at_k = (
            exact_hits / len(relevant_chunk_ids)
            if relevant_chunk_ids
            else 0.0
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        contextual_precision = None
        contextual_recall = None

        # These metrics require an ideal expected answer.
        if expected_output:
            contextual_precision = self._measure_metric(
                metric=ContextualPrecisionMetric(
                    threshold=self.settings.deepeval_threshold,
                    model=self.settings.deepeval_model,
                    include_reason=(
                        self.settings.deepeval_include_reason
                    ),
                    async_mode=False,
                ),
                test_case=test_case,
            )

            contextual_recall = self._measure_metric(
                metric=ContextualRecallMetric(
                    threshold=self.settings.deepeval_threshold,
                    model=self.settings.deepeval_model,
                    include_reason=(
                        self.settings.deepeval_include_reason
                    ),
                    async_mode=False,
                ),
                test_case=test_case,
            )

        faithfulness = self._measure_metric(
            metric=FaithfulnessMetric(
                threshold=self.settings.deepeval_threshold,
                model=self.settings.deepeval_model,
                include_reason=self.settings.deepeval_include_reason,
                async_mode=False,
            ),
            test_case=test_case,
        )

        answer_relevancy = self._measure_metric(
            metric=AnswerRelevancyMetric(
                threshold=self.settings.deepeval_threshold,
                model=self.settings.deepeval_model,
                include_reason=self.settings.deepeval_include_reason,
                async_mode=False,
            ),
            test_case=test_case,
        )

        return EvalItem(
            question=question,
            actual_output=actual_output,
            expected_output=expected_output,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=sorted(relevant_chunk_ids),
            contextual_precision=contextual_precision,
            contextual_recall=contextual_recall,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
        )

    def _generate_answer(
        self,
        question: str,
        retrieved: list,
    ) -> str:
        """
        Generate an answer using the same generator used by /query.

        AnswerGenerator.generate() returns:
            tuple[str, list[Citation]]
        """

        result = self.generator.generate(
            question,
            retrieved,
        )

        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "AnswerGenerator.generate() must return a tuple "
                "in the form (answer, citations)."
            )

        answer, _citations = result

        if not isinstance(answer, str):
            raise TypeError(
                 "AnswerGenerator.generate() returned an invalid answer. "
                f"Expected str, received {type(answer).__name__}."
            )
        
        answer = answer.strip()

        if not answer:
            raise ValueError(
                "AnswerGenerator.generate() returned an empty answer."
        )

        return answer

        
    @staticmethod
    def _format_retrieval_context(result: Any) -> str:
        chunk = result.chunk

        file_path = getattr(chunk, "file_path", "unknown")
        start_line = getattr(chunk, "start_line", None)
        end_line = getattr(chunk, "end_line", None)
        symbol_name = getattr(chunk, "symbol_name", None)
        symbol_type = getattr(chunk, "symbol_type", None)
        chunk_id = getattr(chunk, "chunk_id", "unknown")
        content = getattr(chunk, "content", "")

        metadata = [
            f"chunk_id: {chunk_id}",
            f"file: {file_path}",
        ]

        if start_line is not None and end_line is not None:
            metadata.append(
                f"lines: {start_line}-{end_line}"
            )

        if symbol_name:
            metadata.append(f"symbol: {symbol_name}")

        if symbol_type:
            metadata.append(f"symbol_type: {symbol_type}")

        header = "\n".join(metadata)

        return f"{header}\n\n{content}"

    @staticmethod
    def _measure_metric(
        metric: Any,
        test_case: LLMTestCase,
    ) -> DeepEvalMetricResult:
        try:
            metric.measure(test_case)

            score = (
                float(metric.score)
                if metric.score is not None
                else None
            )

            threshold = float(metric.threshold)

            success = bool(
                score is not None and score >= threshold
            )

            return DeepEvalMetricResult(
                score=score,
                threshold=threshold,
                success=success,
                reason=getattr(metric, "reason", None),
            )

        except Exception as exc:
            return DeepEvalMetricResult(
                score=None,
                threshold=float(metric.threshold),
                success=False,
                reason=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _load_dataset(
        dataset_path: str,
    ) -> list[dict[str, Any]]:
        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {dataset_path}"
            )

        rows: list[dict[str, Any]] = []

        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {dataset_path}:"
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Evaluation row {line_number} must be "
                    "a JSON object."
                )

            rows.append(row)

        if not rows:
            raise ValueError(
                f"No evaluation records found in {dataset_path}"
            )

        return rows

    @staticmethod
    def _required_string(
        row: dict[str, Any],
        field_name: str,
    ) -> str:
        value = row.get(field_name)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Evaluation row requires non-empty "
                f"'{field_name}'."
            )

        return value.strip()

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None

        text = str(value).strip()

        return text or None

    def _build_response(
        self,
        items: list[EvalItem],
    ) -> EvalResponse:
        total = len(items)

        if total == 0:
            return EvalResponse(
                total_items=0,
                mean_context_precision=0.0,
                mean_context_recall=0.0,
                items=[],
            )

        exact_precision = self._mean(
            item.precision_at_k
            for item in items
        )

        exact_recall = self._mean(
            item.recall_at_k
            for item in items
        )

        deepeval_precision = self._mean_optional(
            item.contextual_precision.score
            for item in items
            if item.contextual_precision is not None
        )

        deepeval_recall = self._mean_optional(
            item.contextual_recall.score
            for item in items
            if item.contextual_recall is not None
        )

        faithfulness = self._mean_optional(
            item.faithfulness.score
            for item in items
            if item.faithfulness is not None
        )

        answer_relevancy = self._mean_optional(
            item.answer_relevancy.score
            for item in items
            if item.answer_relevancy is not None
        )

        metric_results = []

        for item in items:
            for result in (
                item.contextual_precision,
                item.contextual_recall,
                item.faithfulness,
                item.answer_relevancy,
            ):
                if result is not None and result.score is not None:
                    metric_results.append(result)

        pass_rate = (
            sum(result.success for result in metric_results)
            / len(metric_results)
            if metric_results
            else None
        )

        return EvalResponse(
            total_items=total,
            mean_context_precision=exact_precision,
            mean_context_recall=exact_recall,
            mean_deepeval_contextual_precision=(
                deepeval_precision
            ),
            mean_deepeval_contextual_recall=deepeval_recall,
            mean_faithfulness=faithfulness,
            mean_answer_relevancy=answer_relevancy,
            pass_rate=pass_rate,
            items=items,
        )

    @staticmethod
    def _mean(values: Any) -> float:
        numbers = list(values)

        if not numbers:
            return 0.0

        return sum(numbers) / len(numbers)

    @staticmethod
    def _mean_optional(
        values: Any,
    ) -> float | None:
        numbers = [
            float(value)
            for value in values
            if value is not None
        ]

        if not numbers:
            return None

        return sum(numbers) / len(numbers)
