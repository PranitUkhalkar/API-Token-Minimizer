from unittest.mock import patch

from src import optimizer


def fake_candidates(prompt):
    half = prompt[: max(1, len(prompt) // 2)]
    return {"conservative": prompt, "balanced": half, "aggressive": half}


def fake_metrics_low_similarity(original, optimized):
    orig_tokens = len(original.split()) or 1
    opt_tokens = len(optimized.split())
    return {
        "original_tokens": orig_tokens,
        "optimized_tokens": opt_tokens,
        "tokens_saved": orig_tokens - opt_tokens,
        "token_reduction_percent": round((orig_tokens - opt_tokens) / orig_tokens * 100, 2),
        "semantic_similarity": 0.5,  # deliberately below the 0.85 threshold
        "resource_savings_note": "note",
    }


def fake_validate_pass(original, candidate):
    return {
        "requirements": [],
        "requirement_score": 1.0,
        "all_hard_constraints_preserved": True,
        "soft_score": 1.0,
        "missing_requirements": [],
        "requirements_found": 0,
        "requirements_preserved": 0,
    }


@patch("src.optimizer.generate_candidates", side_effect=fake_candidates)
@patch("src.optimizer.calculate_metrics", side_effect=fake_metrics_low_similarity)
@patch("src.optimizer.validate_candidate", side_effect=fake_validate_pass)
def test_optimizer_falls_back_to_original_when_similarity_too_low(m1, m2, m3):
    prompt = "This is a test prompt with several words in it for testing."
    result = optimizer.optimize_prompt(prompt)
    assert result["accepted"] is False
    assert result["optimized_prompt"] == prompt
    assert "candidate_failures" in result
    assert any("similarity" in reason for reason in result["candidate_failures"])


def test_optimizer_rejects_empty_prompt():
    result = optimizer.optimize_prompt("   ")
    assert result["accepted"] is False
    assert "Empty prompt" in result["reason"]


def fake_validate_hard_fail(original, candidate):
    return {
        "requirements": [],
        "requirement_score": 0.5,
        "all_hard_constraints_preserved": False,  # a table/number/negation was lost
        "soft_score": 1.0,
        "missing_requirements": [{"text": "use a table", "kind": "format"}],
        "requirements_found": 1,
        "requirements_preserved": 0,
    }


def fake_metrics_high_similarity(original, optimized):
    orig_tokens = len(original.split()) or 1
    opt_tokens = max(1, len(optimized.split()) - 1)
    return {
        "original_tokens": orig_tokens,
        "optimized_tokens": opt_tokens,
        "tokens_saved": orig_tokens - opt_tokens,
        "token_reduction_percent": round((orig_tokens - opt_tokens) / orig_tokens * 100, 2),
        "semantic_similarity": 0.99,
        "resource_savings_note": "note",
    }


@patch("src.optimizer.generate_candidates", side_effect=fake_candidates)
@patch("src.optimizer.calculate_metrics", side_effect=fake_metrics_high_similarity)
@patch("src.optimizer.validate_candidate", side_effect=fake_validate_hard_fail)
def test_optimizer_rejects_high_similarity_candidate_that_drops_a_hard_constraint(m1, m2, m3):
    """A candidate can look great on similarity alone and still be unsafe —
    the hard-constraint gate must reject it regardless."""
    prompt = "Use a comparison table to summarize the results clearly for the reader."
    result = optimizer.optimize_prompt(prompt)
    assert result["accepted"] is False
    assert result["optimized_prompt"] == prompt