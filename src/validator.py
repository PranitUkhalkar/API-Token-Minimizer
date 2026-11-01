"""
Requirement-preservation validator.

TWO FIXES vs. the earlier version:

1. Each requirement is compared against the BEST-MATCHING SENTENCE of the
   optimized prompt (max similarity across candidate sentences), not
   against the whole optimized prompt as one blob. Comparing "one
   requirement" to "the entire prompt" waters down the signal and was
   the root cause of both false accepts and false rejects.

2. Requirement kinds are split into HARD constraints and SOFT-scored
   ones:
       hard  -> quantity, negative, format   (each must individually
                survive; these are never averaged away)
       soft  -> inclusion, general           (scored as a fraction, so
                a safe paraphrase like "avoid unnecessary mathematical
                details" -> "keep mathematics minimal" is correctly
                accepted via semantic similarity + partial term overlap,
                instead of being penalized for not repeating the same
                words)
"""

from __future__ import annotations
from typing import Dict, List

from .requirement_extractor import Requirement, extract_requirements
from .text_utils import split_sentences
from .metrics import embed, cosine_sim

INCLUSION_SIMILARITY_THRESHOLD = 0.55
NEGATIVE_SIMILARITY_THRESHOLD = 0.55
TERM_OVERLAP_THRESHOLD = 0.4  # fraction of a requirement's critical terms that must survive

HARD_KINDS = {"quantity", "negative", "format"}


def _best_sentence_similarity(requirement_text: str, candidate_sentences: List[str]) -> float:
    if not candidate_sentences:
        return 0.0
    texts = [requirement_text] + candidate_sentences
    embeddings = embed(texts)
    req_vec = embeddings[0]
    return max(cosine_sim(req_vec, cand_vec) for cand_vec in embeddings[1:])


def _term_overlap(req: Requirement, candidate_lower: str) -> float:
    if not req.terms:
        return 1.0
    present = sum(1 for t in req.terms if t in candidate_lower)
    return present / len(req.terms)


def _numbers_preserved(req: Requirement, candidate: str) -> bool:
    if not req.numbers:
        return True
    candidate_lower = candidate.lower()
    return all(num.lower() in candidate_lower for num in req.numbers)


def _format_keywords_preserved(req: Requirement, candidate_lower: str) -> bool:
    # Format keywords are not safely paraphrasable ("table" cannot become
    # "structured layout" and still guarantee the model outputs a table),
    # so require literal (case-insensitive) presence.
    # req.terms includes any FORMAT_KEYWORDS hits plus generic content
    # words; only enforce the ones that look like format keywords.
    from .requirement_extractor import FORMAT_KEYWORDS
    relevant = [t for t in req.terms if t in FORMAT_KEYWORDS]
    if not relevant:
        return True
    return all(t in candidate_lower for t in relevant)


def validate_requirement(req: Requirement, candidate: str, candidate_sentences: List[str]) -> Dict:
    candidate_lower = candidate.lower()

    if req.kind == "quantity":
        similarity = _best_sentence_similarity(req.text, candidate_sentences)
        preserved = _numbers_preserved(req, candidate) and _term_overlap(req, candidate_lower) >= TERM_OVERLAP_THRESHOLD

    elif req.kind == "format":
        similarity = _best_sentence_similarity(req.text, candidate_sentences)
        preserved = _format_keywords_preserved(req, candidate_lower)

    elif req.kind == "negative":
        similarity = _best_sentence_similarity(req.text, candidate_sentences)
        preserved = (
            similarity >= NEGATIVE_SIMILARITY_THRESHOLD
            and _term_overlap(req, candidate_lower) >= TERM_OVERLAP_THRESHOLD
        )

    elif req.kind == "inclusion":
        similarity = _best_sentence_similarity(req.text, candidate_sentences)
        preserved = (
            similarity >= INCLUSION_SIMILARITY_THRESHOLD
            and _term_overlap(req, candidate_lower) >= TERM_OVERLAP_THRESHOLD
        )

    else:  # persona / general -> not a hard requirement by definition
        similarity = _best_sentence_similarity(req.text, candidate_sentences)
        preserved = True

    return {
        "text": req.text,
        "kind": req.kind,
        "critical": req.critical,
        "preserved": bool(preserved),
        "similarity": round(float(similarity), 4),
    }


def validate_candidate(original_prompt: str, candidate: str) -> Dict:
    requirements = [r for r in extract_requirements(original_prompt) if r.critical]
    candidate_sentences = split_sentences(candidate)

    results = [validate_requirement(r, candidate, candidate_sentences) for r in requirements]

    hard_results = [r for r in results if r["kind"] in HARD_KINDS]
    soft_results = [r for r in results if r["kind"] not in HARD_KINDS]

    all_hard_preserved = all(r["preserved"] for r in hard_results) if hard_results else True
    soft_score = (
        sum(1 for r in soft_results if r["preserved"]) / len(soft_results)
        if soft_results else 1.0
    )

    total = len(results)
    preserved_count = sum(1 for r in results if r["preserved"])
    requirement_score = (preserved_count / total) if total else 1.0

    missing = [r for r in results if not r["preserved"]]

    return {
        "requirements": results,
        "requirement_score": round(requirement_score, 4),
        "all_hard_constraints_preserved": all_hard_preserved,
        "soft_score": round(soft_score, 4),
        "missing_requirements": missing,
        "requirements_found": total,
        "requirements_preserved": preserved_count,
    }