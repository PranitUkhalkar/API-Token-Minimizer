from unittest.mock import patch

from src import metrics


@patch("src.metrics.semantic_similarity", return_value=0.99)
def test_token_reduction_percent_is_positive_for_shorter_text(mock_sim):
    result = metrics.calculate_metrics(
        "This is a somewhat longer original prompt with extra words in it.",
        "Shorter prompt.",
    )
    assert result["optimized_tokens"] < result["original_tokens"]
    assert result["token_reduction_percent"] > 0
    assert result["tokens_saved"] > 0


@patch("src.metrics.semantic_similarity", return_value=1.0)
def test_identical_text_has_zero_reduction(mock_sim):
    text = "Exactly the same text on both sides."
    result = metrics.calculate_metrics(text, text)
    assert result["token_reduction_percent"] == 0
    assert result["tokens_saved"] == 0


@patch("src.metrics.semantic_similarity", return_value=1.0)
def test_empty_original_does_not_divide_by_zero(mock_sim):
    result = metrics.calculate_metrics("", "")
    assert result["token_reduction_percent"] == 0