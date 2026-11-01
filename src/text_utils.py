"""
Shared text utilities: sentence splitting, whitespace normalization,
and a small, conservative library of safe filler-removal / phrase
simplification rules.

These rules are deliberately narrow. Every entry here is a substitution
that cannot plausibly delete a requirement, a number, a format keyword,
or a negative constraint — that safety property is what lets the
compressor apply them unconditionally, even to "critical" sentences.
"""

import re

FILLER_WORDS = [
    "please", "kindly", "just", "really", "actually",
    "basically", "essentially", "obviously", "of course", "certainly",
]

# (pattern, replacement) — applied in order, case-insensitive.
PHRASE_SIMPLIFICATIONS = [
    (r"\bin order to\b", "to"),
    (r"\bfor the purpose of\b", "to"),
    (r"\bat this point in time\b", "now"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bin the event that\b", "if"),
    (r"\ba wide variety of\b", "various"),
    (r"\bin a simple and easy-to-understand manner\b", "simply"),
    (r"\bin a clear and concise manner\b", "clearly"),
    (r"\bplease note that\b", ""),
    (r"\bit is important to note that\b", ""),
    (r"\bmake sure to\b", ""),
    (r"\bkeep in mind that\b", ""),
    (r"\bas much as possible\b", ""),
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str):
    """Simple, deterministic sentence splitter (splits after . ! ?)."""
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def remove_filler_words(sentence: str) -> str:
    out = sentence
    for w in FILLER_WORDS:
        out = re.sub(rf"\b{re.escape(w)}\b,?", "", out, flags=re.IGNORECASE)
    return normalize_whitespace(out)


def apply_phrase_simplifications(sentence: str) -> str:
    out = sentence
    for pattern, repl in PHRASE_SIMPLIFICATIONS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return normalize_whitespace(out)


def capitalize_first(sentence: str) -> str:
    """Capitalize only the first character; leaves the rest untouched so
    proper nouns / acronyms inside the sentence are never altered."""
    s = sentence.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]