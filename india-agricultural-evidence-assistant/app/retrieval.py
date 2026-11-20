from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime
from functools import lru_cache
from math import exp

from app.models import Chunk, EvidenceDecision, RetrievedChunk
from app.storage import ChromaStore, VectorStore

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "do", "for", "how", "in", "is", "of", "on", "or",
    "the", "to", "what", "when", "with", "would", "i", "my", "me",
}
_CROPS = {
    "rice", "paddy", "wheat", "cotton", "maize", "corn", "sugarcane", "tomato",
    "potato", "soybean", "groundnut", "pulses", "millet", "sorghum", "chickpea",
    "lentil", "mustard",
}
_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
}
_SOIL_PROPERTY_QUERY_TERMS = {
    "pH": re.compile(r"\bph\b", re.IGNORECASE),
    "electrical conductivity": re.compile(r"\b(?:ec|electrical conductivity)\b", re.IGNORECASE),
    "organic carbon": re.compile(r"\borganic carbon\b", re.IGNORECASE),
    "salinity": re.compile(r"\b(?:salin\w*|alkalin\w*)\b", re.IGNORECASE),
}
_SOIL_PROPERTY_EVIDENCE_TERMS = {
    "pH": re.compile(
        r"\bph\b.{0,35}\b(?:\d+(?:\.\d+)?|between|range)\b|\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\s*ph\b",
        re.IGNORECASE,
    ),
    "electrical conductivity": re.compile(r"\b(?:ec|electrical conductivity)\b", re.IGNORECASE),
    "organic carbon": re.compile(r"\borganic carbon\b", re.IGNORECASE),
    "salinity": re.compile(r"\b(?:salin\w*|alkalin\w*)\b", re.IGNORECASE),
}
_RICE_SOIL_QUERY_EXPANSION = "soil texture heavy clay coarse sandy loamy groundwater water input"


def terms(text: str) -> list[str]:
    return [word.lower() for word in _TOKEN.findall(text) if word.lower() not in _STOPWORDS]


def _is_instruction_analysis_question(query: str) -> bool:
    query_terms = set(terms(query))
    return "ignore" in query_terms and bool({"instruction", "instructions", "previous"} & query_terms)


def _asks_agricultural_advice(query: str) -> bool:
    lower_query = query.casefold()
    if _is_instruction_analysis_question(query):
        return False
    advice_request = re.search(
        r"\b(?:should\s+(?:i|we|(?:\w+\s+){0,3}farmers?)|what\s+should|how\s+should|"
        r"how\s+to|how\s+do\s+i|when\s+should|can\s+i|good\s+time\s+to)\b",
        lower_query,
    )
    direct_request = re.search(r"\b(?:please\s+)?(?:recommend|advise)\b", lower_query)
    if not advice_request and not direct_request:
        return False
    action_terms = {
        "apply", "control", "dose", "fertilizer", "fertilize", "fungicide",
        "insecticide", "irrigate", "irrigation", "manage", "pesticide", "spray",
        "treat",
    }
    return bool(action_terms.intersection(terms(query)))


def reciprocal_rank_fusion(
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    constant: int = 60,
    lexical_weight: float = 2.0,
) -> dict[str, float]:
    if constant <= 0 or lexical_weight <= 0:
        raise ValueError("RRF constant and lexical weight must be positive.")
    scores: defaultdict[str, float] = defaultdict(float)
    for rank, (chunk_id, _) in enumerate(dense, start=1):
        scores[chunk_id] += 1.0 / (constant + rank)
    for rank, (chunk_id, _) in enumerate(lexical, start=1):
        scores[chunk_id] += lexical_weight / (constant + rank)
    return dict(scores)


def _matches(chunk: Chunk, filters: dict[str, str]) -> bool:
    for key, wanted in filters.items():
        if wanted and getattr(chunk.metadata, key, "").casefold() != wanted.casefold():
            return False
    return True


def _expand_retrieval_query(query: str) -> str:
    if re.search(r"\b(?:rice|paddy)\b", query, re.IGNORECASE) and re.search(
        r"\bsoils?\b", query, re.IGNORECASE
    ):
        return f"{query} {_RICE_SOIL_QUERY_EXPANSION}"
    return query


def filter_contextual_evidence(
    query: str,
    evidence: list[RetrievedChunk],
    filters: dict[str, str] | None = None,
) -> list[RetrievedChunk]:
    filters = filters or {}
    lower_query = query.casefold()
    asks_advice = _asks_agricultural_advice(query)
    if not asks_advice:
        return evidence

    known_crops = _CROPS | {
        item.chunk.metadata.crop.casefold()
        for item in evidence
        if item.chunk.metadata.crop
    }
    known_places = _STATES | {
        value.casefold()
        for item in evidence
        for value in (item.chunk.metadata.state, item.chunk.metadata.district)
        if value
    }
    query_crops = {crop for crop in known_crops if crop in lower_query}
    query_places = {place for place in known_places if place in lower_query}
    known_seasons = {"kharif", "rabi", "zaid", "summer", "monsoon"} | {
        item.chunk.metadata.season.casefold()
        for item in evidence
        if item.chunk.metadata.season
    }
    query_seasons = {season for season in known_seasons if season in lower_query}

    def matches(item: RetrievedChunk) -> bool:
        metadata = item.chunk.metadata
        if filters.get("crop") and metadata.crop.casefold() != filters["crop"].casefold():
            return False
        if query_crops and not any(
            crop in metadata.crop.casefold() for crop in query_crops
        ):
            return False
        if filters.get("state") and metadata.state.casefold() != filters["state"].casefold():
            return False
        if filters.get("district") and metadata.district.casefold() != filters["district"].casefold():
            return False
        if query_places and not any(
            place in (metadata.state + " " + metadata.district).casefold()
            for place in query_places
        ):
            return False
        if filters.get("season") and metadata.season.casefold() != filters["season"].casefold():
            return False
        if query_seasons and not any(
            season in metadata.season.casefold() for season in query_seasons
        ):
            return False
        return True

    return [item for item in evidence if matches(item)]


def _lexical_ranking(
    query: str,
    chunks: list[Chunk],
    limit: int,
) -> list[tuple[str, float]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        query_terms = set(terms(query))
        scored = [
            (chunk.chunk_id, len(query_terms.intersection(terms(chunk.text))) / max(1, len(query_terms)))
            for chunk in chunks
        ]
    else:
        corpus = [terms(chunk.text) for chunk in chunks]
        if not any(corpus):
            return []
        scores = BM25Okapi(corpus).get_scores(terms(query))
        maximum = max(scores, default=0.0)
        scored = [
            (chunk.chunk_id, float(score / maximum) if maximum > 0 else 0.0)
            for chunk, score in zip(chunks, scores)
        ]
    return sorted(
        (item for item in scored if item[1] > 0),
        key=lambda item: (-item[1], item[0]),
    )[:limit]


def _dense_ranking(
    store: VectorStore,
    query: str,
    limit: int,
    framework: str,
) -> list[tuple[str, float]]:
    if isinstance(store, ChromaStore):
        if framework == "llamaindex":
            from app.frameworks.llamaindex_adapter import retrieve
        else:
            from app.frameworks.langchain_adapter import retrieve
        return [
            (chunk.chunk_id, score)
            for chunk, score in retrieve(store, query, limit)
        ]
    return store.search(query, limit)


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    model_name: str,
) -> tuple[list[RetrievedChunk], str]:
    if not model_name:
        return candidates, "Disabled by configuration."
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return candidates, "Unavailable: install sentence-transformers."
    try:
        model = _cross_encoder(model_name)
        scores = model.predict([(query, item.chunk.text) for item in candidates])
    except (OSError, RuntimeError, ValueError) as error:
        return candidates, f"Unavailable: {error}"
    for item, score in zip(candidates, scores):
        raw_score = max(-60.0, min(60.0, float(score)))
        item.reranker_score = 1.0 / (1.0 + exp(-raw_score))
    return sorted(
        candidates,
        key=lambda item: (item.reranker_score or 0.0, item.fusion_score),
        reverse=True,
    ), f"Enabled ({model_name})."


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def is_current_official(metadata, max_age_days: int, today: date | None = None) -> bool:
    if metadata.status.casefold() not in {"official", "official source"} or not metadata.issued_date:
        return False
    try:
        issued = datetime.fromisoformat(metadata.issued_date[:10]).date()
    except ValueError:
        return False
    reference_day = today or date.today()
    age_days = (reference_day - issued).days
    return 0 <= age_days <= max_age_days


def _advisory_is_valid_today(metadata, today: date | None = None) -> bool:
    if "advisory" not in metadata.document_type.casefold() or not metadata.validity_end:
        return False
    try:
        validity_end = date.fromisoformat(metadata.validity_end[:10])
        validity_start = (
            date.fromisoformat(metadata.validity_start[:10])
            if metadata.validity_start
            else None
        )
    except ValueError:
        return False
    reference_day = today or date.today()
    return (validity_start is None or validity_start <= reference_day) and reference_day <= validity_end


def retrieve(
    store: VectorStore,
    query: str,
    framework: str,
    dense_limit: int,
    lexical_limit: int,
    rerank_limit: int,
    reranker_model: str,
    filters: dict[str, str] | None = None,
) -> tuple[list[RetrievedChunk], dict[str, object]]:
    filters = filters or {}
    all_chunks = [chunk for chunk in store.all_chunks() if _matches(chunk, filters)]
    allowed = {chunk.chunk_id: chunk for chunk in all_chunks}
    if not allowed:
        return [], {
            "dense": [],
            "lexical": [],
            "fusion": {},
            "reranker_state": "Not run; no chunks matched the filters.",
            "query_expansion": "",
        }

    retrieval_query = _expand_retrieval_query(query)
    dense_candidates = [
        item for item in _dense_ranking(
            store,
            retrieval_query,
            min(len(all_chunks), dense_limit * 4),
            framework,
        )
        if item[0] in allowed
    ][:dense_limit]
    lexical_candidates = _lexical_ranking(retrieval_query, all_chunks, lexical_limit)
    fusion = reciprocal_rank_fusion(dense_candidates, lexical_candidates)
    ordered_ids = sorted(fusion, key=lambda chunk_id: (-fusion[chunk_id], chunk_id))
    candidates = [
        RetrievedChunk(
            chunk=allowed[chunk_id],
            dense_score=next((score for identifier, score in dense_candidates if identifier == chunk_id), 0.0),
            lexical_score=next((score for identifier, score in lexical_candidates if identifier == chunk_id), 0.0),
            fusion_score=fusion[chunk_id],
        )
        for chunk_id in ordered_ids[:rerank_limit]
    ]
    ranked, reranker_state = rerank(retrieval_query, candidates, reranker_model)
    return ranked, {
        "dense": [
            {
                "chunk_id": chunk_id,
                "source_id": allowed[chunk_id].metadata.source_id,
                "title": allowed[chunk_id].metadata.title,
                "page_number": allowed[chunk_id].page_number,
                "score": score,
            }
            for chunk_id, score in dense_candidates
        ],
        "lexical": [
            {
                "chunk_id": chunk_id,
                "source_id": allowed[chunk_id].metadata.source_id,
                "title": allowed[chunk_id].metadata.title,
                "page_number": allowed[chunk_id].page_number,
                "score": score,
            }
            for chunk_id, score in lexical_candidates
            if chunk_id in allowed
        ],
        "fusion": fusion,
        "reranker_state": reranker_state,
        "query_expansion": (
            retrieval_query[len(query):].strip()
            if retrieval_query != query
            else ""
        ),
    }


def evidence_gate(
    query: str,
    evidence: list[RetrievedChunk],
    minimum_score: float,
    minimum_coverage: float,
    filters: dict[str, str] | None = None,
    max_official_source_age_days: int = 365,
    require_current_advisory: bool = False,
) -> EvidenceDecision:
    filters = filters or {}
    question_terms = set(terms(query))
    evidence_terms = set(terms(" ".join(item.chunk.text for item in evidence[:4])))
    coverage = len(question_terms & evidence_terms) / max(1, len(question_terms))
    best_score = max(
        (
            item.reranker_score
            if item.reranker_score is not None
            else max(item.dense_score, item.lexical_score)
            for item in evidence
        ),
        default=0.0,
    )
    lower_query = query.casefold()
    instruction_test = _is_instruction_analysis_question(query)
    asks_advice = _asks_agricultural_advice(query) or require_current_advisory
    missing: list[str] = []
    if require_current_advisory:
        current_advisories = [
            item
            for item in evidence
            if "advisory" in item.chunk.metadata.document_type.casefold()
            and item.chunk.metadata.status.casefold() in {"official", "official source"}
            and _advisory_is_valid_today(item.chunk.metadata)
        ]
        if not current_advisories:
            return EvidenceDecision(
                False,
                "No current official advisory with an active validity window was retrieved.",
                coverage,
                best_score,
                ("current official advisory",),
            )
    if asks_advice:
        advisory_evidence = [
            item
            for item in evidence
            if "advisory" in item.chunk.metadata.document_type.casefold()
        ]
        if advisory_evidence and not any(
            _advisory_is_valid_today(item.chunk.metadata) for item in advisory_evidence
        ):
            return EvidenceDecision(
                False,
                "Retrieved advisory material is expired or lacks a usable validity window for current advice.",
                coverage,
                best_score,
                ("current advisory validity",),
            )
        crops_in_evidence = {
            item.chunk.metadata.crop.casefold()
            for item in evidence[:3]
            if item.chunk.metadata.crop
        }
        known_crops = _CROPS | crops_in_evidence
        crops_in_query = {crop for crop in known_crops if crop in lower_query}
        if not filters.get("crop") and not crops_in_query:
            missing.append("crop")
        elif not crops_in_evidence:
            missing.append("crop metadata")
        elif crops_in_query and not any(
            query_crop in source_crop or source_crop in query_crop
            for query_crop in crops_in_query
            for source_crop in crops_in_evidence
        ):
            return EvidenceDecision(
                False,
                "Retrieved passages do not match the crop named in the question.",
                coverage,
                best_score,
                ("matching crop evidence",),
            )
        places_in_evidence = {
            value.casefold()
            for item in evidence[:3]
            for value in (item.chunk.metadata.state, item.chunk.metadata.district)
            if value
        }
        places_in_query = {
            place for place in places_in_evidence | _STATES if place in lower_query
        }
        if not (filters.get("state") or filters.get("district")) and not places_in_query:
            missing.append("location")
        elif not places_in_evidence:
            missing.append("location metadata")
        elif places_in_query and not any(
            query_place in source_place or source_place in query_place
            for query_place in places_in_query
            for source_place in places_in_evidence
        ):
            return EvidenceDecision(
                False,
                "Retrieved passages do not match the location named in the question.",
                coverage,
                best_score,
                ("matching location evidence",),
            )
        seasons_in_evidence = {
            item.chunk.metadata.season.casefold()
            for item in evidence[:3]
            if item.chunk.metadata.season
        }
        known_seasons = {"kharif", "rabi", "zaid", "summer", "monsoon"} | seasons_in_evidence
        seasons_in_query = {season for season in known_seasons if season in lower_query}
        if not filters.get("season") and not seasons_in_query:
            missing.append("season")
        elif not seasons_in_evidence:
            missing.append("season metadata")
        elif seasons_in_query and not any(
            season in evidence_season
            for season in seasons_in_query
            for evidence_season in seasons_in_evidence
        ):
            return EvidenceDecision(
                False,
                "Retrieved passages do not match the season named in the question.",
                coverage,
                best_score,
                ("matching season evidence",),
            )
    if missing:
        return EvidenceDecision(
            False,
            "This advisory question needs more context before evidence can be applied.",
            coverage,
            best_score,
            tuple(missing),
        )
    if not evidence:
        return EvidenceDecision(False, "No documents matched the question and filters.", 0.0, 0.0)
    requested_soil_properties = [
        label
        for label, pattern in _SOIL_PROPERTY_QUERY_TERMS.items()
        if pattern.search(query)
    ]
    unsupported_soil_properties = [
        label
        for label in requested_soil_properties
        if not any(
            _SOIL_PROPERTY_EVIDENCE_TERMS[label].search(item.chunk.text)
            for item in evidence[:4]
        )
    ]
    if unsupported_soil_properties:
        return EvidenceDecision(
            False,
            "The corpus does not contain evidence for the specific soil property requested.",
            coverage,
            best_score,
            tuple(f"source covering {property_name}" for property_name in unsupported_soil_properties),
        )
    if not instruction_test and {
        "pesticide", "insecticide", "fungicide", "dose"
    }.intersection(terms(query)):
        official_evidence = [
            item
            for item in evidence
            if is_current_official(item.chunk.metadata, max_official_source_age_days)
        ]
        official_current = bool(official_evidence)
        if not official_current:
            return EvidenceDecision(
                False,
                "Pesticide-related guidance requires a current official source; the retrieved corpus does not provide one.",
                coverage,
                best_score,
            )
        asks_for_rate = any(word in lower_query for word in ("dose", "rate", "how much", "application"))
        supported_rate = re.search(
            r"\b\d+(?:\.\d+)?\s*(?:kg|g|ml|l|litre|liter|ha|acre|%)\b",
            " ".join(item.chunk.text.casefold() for item in official_evidence),
        )
        if asks_for_rate and not supported_rate:
            return EvidenceDecision(
                False,
                "The current official evidence does not explicitly support a pesticide rate or dose.",
                coverage,
                best_score,
            )
    conflicts = (
        ({"increase", "higher", "more", "raise"}, {"decrease", "lower", "less", "reduce"}),
        ({"apply", "use", "irrigate"}, {"avoid", "do not", "stop"}),
    )
    grouped: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    query_tokens = set(terms(query))
    query_crops = {
        crop
        for crop in _CROPS
        | {
            item.chunk.metadata.crop.casefold()
            for item in evidence
            if item.chunk.metadata.crop
        }
        if crop in lower_query
    }
    query_places = {
        place
        for place in _STATES
        | {
            value.casefold()
            for item in evidence
            for value in (item.chunk.metadata.state, item.chunk.metadata.district)
            if value
        }
        if place in lower_query
    }
    query_seasons = {
        season
        for season in {"kharif", "rabi", "zaid", "summer", "monsoon"}
        | {
            item.chunk.metadata.season.casefold()
            for item in evidence
            if item.chunk.metadata.season
        }
        if season in lower_query
    }
    if asks_advice and not instruction_test:
        for item in evidence[:5]:
            metadata = item.chunk.metadata
            if not metadata.crop or (not metadata.state and not metadata.district):
                continue
            if filters.get("crop") and metadata.crop.casefold() != filters["crop"].casefold():
                continue
            if query_crops and not any(
                crop in metadata.crop.casefold() for crop in query_crops
            ):
                continue
            places = (metadata.state + " " + metadata.district).casefold()
            if filters.get("state") and metadata.state.casefold() != filters["state"].casefold():
                continue
            if filters.get("district") and metadata.district.casefold() != filters["district"].casefold():
                continue
            if query_places and not any(place in places for place in query_places):
                continue
            if filters.get("season") and metadata.season.casefold() != filters["season"].casefold():
                continue
            if query_seasons and not any(
                season in metadata.season.casefold() for season in query_seasons
            ):
                continue
            overlap = len(query_tokens.intersection(terms(item.chunk.text))) / max(1, len(query_tokens))
            if overlap < 0.15:
                continue
            key = (metadata.crop.casefold(), metadata.state.casefold(), metadata.district.casefold())
            grouped[key].append(item.chunk.text.casefold())
    has_conflict = False
    if asks_advice and not instruction_test:
        for texts in grouped.values():
            if len(texts) < 2:
                continue
            for left, right in conflicts:
                left_sources = [any(term in text for term in left) for text in texts]
                right_sources = [any(term in text for term in right) for text in texts]
                has_conflict |= any(a and b for index, a in enumerate(left_sources) for b in right_sources[index + 1 :])
                has_conflict |= any(b and a for index, b in enumerate(right_sources) for a in left_sources[index + 1 :])
    if has_conflict:
        return EvidenceDecision(
            False,
            "Retrieved sources contain opposing guidance that this prototype cannot resolve.",
            coverage,
            best_score,
        )
    if coverage < minimum_coverage:
        return EvidenceDecision(False, "Retrieved passages do not cover enough of the question.", coverage, best_score)
    if best_score < minimum_score:
        return EvidenceDecision(False, "The strongest retrieval match is below the configured threshold.", coverage, best_score)
    return EvidenceDecision(True, "Evidence passed the deterministic relevance and coverage checks.", coverage, best_score)
