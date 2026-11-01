"""
Requirement / Constraint Extraction
------------------------------------
Deterministic, regex + heuristic extraction of everything a prompt
requires of the model's answer: things that MUST survive compression.

This module intentionally contains NO machine learning. Regex-based
extraction is fully explainable and auditable — exactly what a
"never silently drop what the user asked for" safety layer needs to be.

Each sentence of the prompt becomes exactly one Requirement, tagged with
a `kind` and a `critical` flag:

    quantity   -> critical  ("at least three examples", "exactly 5 steps")
    negative   -> critical  ("avoid ...", "do not ...", "without ...")
    format     -> critical  ("use a table", "clear headings", "output JSON")
    inclusion  -> critical  ("explain X", "include Y", "compare A and B")
    persona    -> NOT critical (scene-setting: "You are an expert ...")
    general    -> NOT critical (anything else / connective sentences)

Only `critical=True` sentences are hard constraints the optimizer must
protect. `persona` and `general` sentences are exactly what the
compressor is allowed to shorten, merge, or drop.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Set

from .text_utils import split_sentences

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_NUM_WORD_ALT = "|".join(NUMBER_WORDS.keys())

QUANTITY_PATTERN = re.compile(
    rf"\b(at least|at most|exactly|no more than|no fewer than|"
    rf"a minimum of|a maximum of|minimum of|maximum of)\s+"
    rf"(\d+|{_NUM_WORD_ALT})\b",
    re.IGNORECASE,
)

NEGATIVE_PATTERN = re.compile(
    r"\b(do not|don't|avoid|without|refrain from|must not|should not|"
    r"never|exclude|skip)\b",
    re.IGNORECASE,
)

# Longer/more specific phrases are listed first so they are preferred
# when a sentence is scanned for which keyword(s) it contains.
FORMAT_KEYWORDS = [
    "comparison table", "bulleted list", "numbered list", "bullet points",
    "bullet point", "step-by-step", "step by step", "code block",
    "outline format", "table", "headings", "heading", "sub-heading",
    "subheading", "json", "csv", "yaml", "markdown", "header",
    "columns", "rows", "checklist",
]

INCLUSION_VERBS = [
    "include", "explain", "describe", "discuss", "compare", "list",
    "analyze", "analyse", "calculate", "summarize", "summarise",
    "outline", "define", "illustrate", "provide", "demonstrate",
    "cover", "address", "mention", "identify", "organize", "organise",
    "ensure", "highlight", "show", "state", "specify",
]

PERSONA_PATTERN = re.compile(
    r"^(you are|act as|please act as|imagine you are|as an?\s)",
    re.IGNORECASE,
)

STOPWORDS_FOR_TERMS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "is",
    "with", "using", "use", "please", "at", "least", "exactly", "when",
    "answer", "prompt", "each", "that", "this", "your", "you", "are",
}


@dataclass
class Requirement:
    text: str                              # the originating sentence
    kind: str                              # quantity|negative|format|inclusion|persona|general
    critical: bool                         # hard constraint if True
    terms: Set[str] = field(default_factory=set)
    numbers: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "critical": self.critical,
            "terms": sorted(self.terms),
            "numbers": self.numbers,
        }


def _extract_terms(sentence: str) -> Set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", sentence.lower())
    return {w for w in words if w not in STOPWORDS_FOR_TERMS}


def _extract_numbers(sentence: str) -> List[str]:
    found: List[str] = []
    for m in QUANTITY_PATTERN.finditer(sentence):
        val = m.group(2).lower()
        if val not in found:
            found.append(val)
    for m in re.finditer(r"\b\d+\b", sentence):
        if m.group(0) not in found:
            found.append(m.group(0))
    return found


def _format_hits(sentence_lower: str) -> List[str]:
    hits = []
    for kw in FORMAT_KEYWORDS:
        if kw in sentence_lower and kw not in hits:
            # avoid double-counting e.g. "table" when "comparison table" already hit
            if not any(kw != other and kw in other for other in hits):
                hits.append(kw)
    return hits


def classify_sentence(sentence: str) -> Requirement:
    s = sentence.strip()
    if not s:
        return Requirement(text=s, kind="empty", critical=False)

    s_lower = s.lower()
    is_negative = bool(NEGATIVE_PATTERN.search(s))
    has_quantity = bool(QUANTITY_PATTERN.search(s))
    format_hits = _format_hits(s_lower)
    has_inclusion_verb = any(
        re.search(rf"\b{re.escape(v)}\b", s, re.IGNORECASE) for v in INCLUSION_VERBS
    )
    is_persona = bool(PERSONA_PATTERN.match(s)) and not (is_negative or has_quantity or format_hits)

    if is_negative:
        kind, critical = "negative", True
    elif has_quantity:
        kind, critical = "quantity", True
    elif format_hits:
        kind, critical = "format", True
    elif has_inclusion_verb:
        kind, critical = "inclusion", True
    elif is_persona:
        kind, critical = "persona", False
    else:
        kind, critical = "general", False

    req = Requirement(
        text=s,
        kind=kind,
        critical=critical,
        terms=_extract_terms(s),
        numbers=_extract_numbers(s),
    )
    if format_hits:
        req.terms |= set(format_hits)
    return req


def extract_requirements(prompt: str) -> List[Requirement]:
    """One Requirement per sentence of the prompt, in original order."""
    return [classify_sentence(s) for s in split_sentences(prompt) if s.strip()]


def critical_requirements(prompt: str) -> List[Requirement]:
    return [r for r in extract_requirements(prompt) if r.critical]
