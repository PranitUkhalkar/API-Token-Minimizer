from src.compressor import generate_candidates

PROMPT = (
    "You are an experienced data scientist and machine learning mentor. "
    "Please explain supervised learning to a beginner in a simple and "
    "easy-to-understand manner. Include the definition, difference between "
    "classification and regression, at least three real-world examples, "
    "the purpose of training and testing datasets, common machine learning "
    "algorithms, and explain when each algorithm is useful. Organize the "
    "answer using clear headings and a comparison table. Avoid unnecessary "
    "mathematical details."
)


def test_conservative_preserves_all_critical_content():
    candidates = generate_candidates(PROMPT)
    conservative = candidates["conservative"].lower()
    assert "three" in conservative
    assert "comparison table" in conservative
    assert "mathematical details" in conservative


def test_balanced_still_preserves_hard_requirements():
    candidates = generate_candidates(PROMPT)
    balanced = candidates["balanced"].lower()
    assert "three" in balanced
    assert "comparison table" in balanced
    assert "mathematical details" in balanced


def test_deterministic_candidates_never_exceed_original_word_count():
    candidates = generate_candidates(PROMPT)
    original_len = len(PROMPT.split())
    assert len(candidates["conservative"].split()) <= original_len
    assert len(candidates["balanced"].split()) <= original_len


def test_balanced_is_not_longer_than_conservative():
    candidates = generate_candidates(PROMPT)
    assert len(candidates["balanced"].split()) <= len(candidates["conservative"].split())