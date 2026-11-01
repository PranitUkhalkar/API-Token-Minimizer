"""
Optimizer: generate candidates, evaluate all of them, pick the best SAFE
one. Never hides a failure.

Objective:
    maximize token_reduction
    subject to:
        optimized_tokens < original_tokens
        whole-prompt semantic_similarity >= SIMILARITY_THRESHOLD
        ALL hard requirement constraints preserved (quantity/negative/format)
        soft requirement score (inclusion/general) >= SOFT_REQUIREMENT_THRESHOLD

If no candidate satisfies every constraint, the ORIGINAL prompt is
returned unchanged, and exactly which constraint each candidate failed
is reported (candidate_failures) — nothing is swept under the rug.

Thresholds are deliberately NOT tuned down to inflate the acceptance
rate. If the acceptance rate is low, the fix belongs in compressor.py
(better candidates) or requirement_extractor.py (better detection),
never here.
"""

from __future__ import annotations
from typing import Dict

from .compressor import generate_candidates
from .validator import validate_candidate
from .metrics import calculate_metrics

SIMILARITY_THRESHOLD = 0.85
SOFT_REQUIREMENT_THRESHOLD = 0.85


def _evaluate_candidate(original: str, candidate: str) -> Dict:
    metrics = calculate_metrics(original, candidate)
    validation = validate_candidate(original, candidate)

    passes = (
        metrics["optimized_tokens"] < metrics["original_tokens"]
        and metrics["semantic_similarity"] >= SIMILARITY_THRESHOLD
        and validation["all_hard_constraints_preserved"]
        and validation["soft_score"] >= SOFT_REQUIREMENT_THRESHOLD
    )

    return {**metrics, **validation, "candidate": candidate, "passes_safety": passes}


def _failure_reasons(ev: Dict) -> list:
    reasons = []
    if ev["optimized_tokens"] >= ev["original_tokens"]:
        reasons.append("no token reduction")
    if ev["semantic_similarity"] < SIMILARITY_THRESHOLD:
        reasons.append(f"semantic similarity {ev['semantic_similarity']} < {SIMILARITY_THRESHOLD}")
    if not ev["all_hard_constraints_preserved"]:
        reasons.append("a hard constraint (quantity/negative/format) was lost")
    if ev["soft_score"] < SOFT_REQUIREMENT_THRESHOLD:
        reasons.append(f"soft requirement score {ev['soft_score']} < {SOFT_REQUIREMENT_THRESHOLD}")
    return reasons


def optimize_prompt(prompt: str) -> Dict:
    if not prompt or not prompt.strip():
        return {
            "accepted": False,
            "reason": "Empty prompt provided.",
            "optimized_prompt": prompt,
            "strategies": {},
        }

    candidates = generate_candidates(prompt)
    evaluations = {name: _evaluate_candidate(prompt, text) for name, text in candidates.items()}

    passing = {name: ev for name, ev in evaluations.items() if ev["passes_safety"]}

    if passing:
        best_name = max(passing, key=lambda n: passing[n]["token_reduction_percent"])
        best = passing[best_name]
        return {
            "accepted": True,
            "strategy_used": best_name,
            "optimized_prompt": best["candidate"],
            "reason": f"Candidate '{best_name}' passed all safety checks with the highest token reduction.",
            "metrics": {
                k: best[k]
                for k in (
                    "original_tokens", "optimized_tokens", "tokens_saved",
                    "token_reduction_percent", "semantic_similarity",
                    "resource_savings_note",
                )
            },
            "requirement_score": best["requirement_score"],
            "requirements": best["requirements"],
            "missing_requirements": best["missing_requirements"],
            "strategies": evaluations,
        }

    # Nothing passed -> fall back to the original prompt, unchanged.
    failure_reasons = [
        f"{name}: {', '.join(_failure_reasons(ev)) or 'unknown'}"
        for name, ev in evaluations.items()
    ]
    original_metrics = calculate_metrics(prompt, prompt)

    return {
        "accepted": False,
        "strategy_used": None,
        "optimized_prompt": prompt,
        "reason": "No generated candidate passed all safety checks; original prompt returned unchanged.",
        "candidate_failures": failure_reasons,
        "metrics": {
            "original_tokens": original_metrics["original_tokens"],
            "optimized_tokens": original_metrics["original_tokens"],
            "tokens_saved": 0,
            "token_reduction_percent": 0.0,
            "semantic_similarity": 1.0,
            "resource_savings_note": original_metrics["resource_savings_note"],
        },
        "strategies": evaluations,
    }