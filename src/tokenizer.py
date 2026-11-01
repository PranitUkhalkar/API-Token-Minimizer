"""
Token counting.

Primary: tiktoken's gpt-4o-mini encoding (matches real OpenAI-style
token billing).

Fallback: a deterministic word/punctuation-unit counter, used ONLY if
tiktoken's encoding files cannot be loaded (e.g. no internet access on
first run, or a fully offline environment). This keeps the rest of the
pipeline testable offline; for the real benchmark numbers you report,
make sure tiktoken is actually working (see README).
"""

import re
import warnings

_encoding = None
_fallback_warned = False


def _get_encoding():
    global _encoding
    if _encoding is not None:
        return _encoding
    try:
        import tiktoken
        _encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception as e:  # pragma: no cover - environment dependent
        _encoding = False
        global _fallback_warned
        if not _fallback_warned:
            warnings.warn(
                f"tiktoken unavailable ({e}); using an approximate "
                "word-based token counter instead. Fix internet access "
                "and re-run for accurate, reportable token counts."
            )
            _fallback_warned = True
    return _encoding


def _approximate_token_count(text: str) -> int:
    units = re.findall(r"\w+|[^\w\s]", text)
    return len(units)


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    if not text:
        return 0
    enc = _get_encoding()
    if enc:
        return len(enc.encode(text))
    return _approximate_token_count(text)


if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    print(f"Token count: {count_tokens(prompt)}")