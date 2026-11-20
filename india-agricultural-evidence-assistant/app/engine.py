from __future__ import annotations

from app.answer import answer_question
from app.config import Settings
from app.ingest import create_store
from app.models import AnswerResult
from app.retrieval import filter_contextual_evidence, retrieve


def ask(
    question: str,
    config: Settings,
    filters: dict[str, str] | None = None,
    require_current_advisory: bool = False,
) -> AnswerResult:
    store = create_store(config)
    evidence, diagnostics = retrieve(
        store=store,
        query=question,
        framework=config.framework,
        dense_limit=config.dense_top_k,
        lexical_limit=config.lexical_top_k,
        rerank_limit=config.rerank_top_k,
        reranker_model=config.reranker_model,
        filters=filters,
    )
    answer_evidence = filter_contextual_evidence(question, evidence, filters)
    result = answer_question(
        question,
        answer_evidence,
        config,
        config.evidence_min_score,
        config.evidence_min_coverage,
        filters,
        config.max_official_source_age_days,
        require_current_advisory,
    )
    result.diagnostics.update(diagnostics)
    return result
