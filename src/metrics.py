"""
Metrics: token counts + semantic similarity (all-MiniLM-L6-v2 embeddings)
+ an honestly-worded resource-savings note.

The embedding model is loaded lazily (once) so that importing this
module never requires a network call or GPU/CPU load until it's
actually needed.
"""

from .tokenizer import count_tokens

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def embed(texts):
    """Return a list of embedding vectors for the given texts."""
    model = _get_embedding_model()
    return model.encode(list(texts))


def cosine_sim(vec_a, vec_b) -> float:
    import numpy as np
    a, b = np.array(vec_a, dtype=float), np.array(vec_b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_similarity(text_a: str, text_b: str) -> float:
    emb = embed([text_a, text_b])
    return round(cosine_sim(emb[0], emb[1]), 4)


# IMPORTANT PROJECT CLAIM (do not change the wording below without
# re-reading section 3 of the project spec): we never claim measured
# carbon savings, only an estimated proxy based on token reduction.
RESOURCE_NOTE = (
    "Estimated inference-resource savings from reduced token usage "
    "(a proxy for reduced compute/API cost, not a measured energy or "
    "carbon-emissions figure)."
)


def calculate_metrics(original: str, optimized: str) -> dict:
    original = original if isinstance(original, str) else str(original)
    optimized = optimized if isinstance(optimized, str) else str(optimized)

    original_tokens = count_tokens(original)
    optimized_tokens = count_tokens(optimized)
    tokens_saved = original_tokens - optimized_tokens

    reduction = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0.0
    similarity = semantic_similarity(original, optimized)

    return {
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "tokens_saved": tokens_saved,
        "token_reduction_percent": round(reduction, 2),
        "semantic_similarity": similarity,
        "resource_savings_note": RESOURCE_NOTE,
    }