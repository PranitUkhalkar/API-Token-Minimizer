"""
These tests mock `src.validator.embed` with a crude deterministic
word-overlap "embedding" so the validator's LOGIC (per-sentence best
match, hard-vs-soft gating, term overlap, number/format checks) can be
verified without downloading the real all-MiniLM-L6-v2 model. Run the
actual project (see README) to exercise the real embeddings.
"""

from unittest.mock import patch

from src import validator


def fake_embed(texts):
    vocab = sorted(set(w.lower() for t in texts for w in t.split()))
    return [[1.0 if w in t.lower().split() else 0.0 for w in vocab] for t in texts]


@patch("src.validator.embed", side_effect=fake_embed)
def test_quantity_requirement_fails_if_number_dropped(mock_embed):
    original = "Include at least three real-world examples."
    candidate_ok = "Include at least three real-world examples."
    candidate_bad = "Include some real-world examples."

    result_ok = validator.validate_candidate(original, candidate_ok)
    result_bad = validator.validate_candidate(original, candidate_bad)

    assert result_ok["all_hard_constraints_preserved"] is True
    assert result_bad["all_hard_constraints_preserved"] is False


@patch("src.validator.embed", side_effect=fake_embed)
def test_format_requirement_requires_literal_keyword(mock_embed):
    original = "Organize the answer using clear headings and a comparison table."
    candidate_missing_table = "Organize the answer using clear headings."
    candidate_ok = "Use headings and a comparison table for the answer."

    bad = validator.validate_candidate(original, candidate_missing_table)
    ok = validator.validate_candidate(original, candidate_ok)

    assert bad["all_hard_constraints_preserved"] is False
    assert ok["all_hard_constraints_preserved"] is True


@patch("src.validator.embed", side_effect=fake_embed)
def test_negative_constraint_must_survive(mock_embed):
    original = "Avoid unnecessary mathematical details."
    candidate_dropped = "Explain the algorithm clearly."
    candidate_kept = "Avoid unnecessary mathematical details in the explanation."

    dropped = validator.validate_candidate(original, candidate_dropped)
    kept = validator.validate_candidate(original, candidate_kept)

    assert dropped["all_hard_constraints_preserved"] is False
    assert kept["all_hard_constraints_preserved"] is True


@patch("src.validator.embed", side_effect=fake_embed)
def test_full_prompt_missing_requirements_are_reported(mock_embed):
    original = (
        "Include at least three examples. Avoid unnecessary jargon. "
        "Use a comparison table."
    )
    candidate = "Include at least three examples."

    result = validator.validate_candidate(original, candidate)
    assert result["all_hard_constraints_preserved"] is False
    assert len(result["missing_requirements"]) >= 1