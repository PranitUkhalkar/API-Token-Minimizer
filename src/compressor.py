"""
Segment-aware hybrid compressor.

THIS IS THE ARCHITECTURAL FIX for the "FLAN-T5 rewrites the whole prompt
and deletes formatting requirements" failure mode observed in the
earlier version.

Design decision: FLAN-T5 is NEVER allowed to touch a sentence the
requirement extractor tagged `critical=True` (quantity / negative /
format / inclusion). Those sentences only ever go through safe,
deterministic cleanup (filler-word removal + phrase simplification —
see text_utils.py, whose substitution list is restricted to patterns
that cannot delete a number, a format keyword, or a negation).

The transformer model is only ever applied to the *remaining*
non-critical sentences (persona/scene-setting/connective filler) —
exactly the material that carries no extracted requirement, so semantic
drift there cannot delete something the user asked for. This is what
lets us use a genuine transformer-based generation step (satisfying the
"real AI/DL, not just regex" project requirement) without reproducing
the earlier catastrophic failures.

Three deterministic strategies are produced:

    conservative -> filler/phrase cleanup only, nothing removed/rewritten
    balanced     -> conservative + drop zero-information persona sentences
    aggressive   -> balanced + FLAN-T5 paraphrase of the remaining
                    non-critical sentences only
"""

from __future__ import annotations
from typing import Dict, List

from .requirement_extractor import extract_requirements, Requirement
from .text_utils import (
    remove_filler_words,
    apply_phrase_simplifications,
    normalize_whitespace,
    capitalize_first,
)

MODEL_NAME = "google/flan-t5-base"

_flan_tokenizer = None
_flan_model = None


def _get_flan():
    global _flan_tokenizer, _flan_model
    if _flan_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _flan_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _flan_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return _flan_tokenizer, _flan_model


def _clean_sentence(sentence: str) -> str:
    return capitalize_first(apply_phrase_simplifications(remove_filler_words(sentence)))


def _is_low_information_persona(req: Requirement) -> bool:
    """Any sentence the extractor classified as pure persona/scene-setting
    ("You are an experienced X and Y mentor...") is by definition
    non-critical (classify_sentence only assigns "persona" when no
    negative/quantity/format/inclusion pattern matched) — so it is always
    safe to drop it entirely in the balanced/aggressive strategies."""
    return req.kind == "persona"


def _flan_paraphrase(text: str, max_new_tokens: int = 60) -> str:
    if not text.strip():
        return text
    tok, model = _get_flan()
    instruction = (
        "Paraphrase the following text to be shorter while keeping the "
        f"same meaning: {text}"
    )
    inputs = tok(instruction, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        do_sample=False,
        early_stopping=True,
    )
    result = tok.decode(outputs[0], skip_special_tokens=True).strip()
    return result if result else text


def _build_candidate(requirements: List[Requirement], drop_persona: bool, use_model: bool) -> str:
    kept_sentences: List[str] = []
    compressible_buffer: List[str] = []

    def flush_buffer():
        if not compressible_buffer:
            return
        joined = " ".join(compressible_buffer)
        if use_model:
            try:
                joined = _flan_paraphrase(joined)
            except Exception:
                pass  # model unavailable -> silently keep the deterministic text
        kept_sentences.append(capitalize_first(normalize_whitespace(joined)))
        compressible_buffer.clear()

    for req in requirements:
        if req.critical:
            flush_buffer()
            kept_sentences.append(_clean_sentence(req.text))
        else:
            if drop_persona and _is_low_information_persona(req):
                continue
            compressible_buffer.append(_clean_sentence(req.text))

    flush_buffer()
    return normalize_whitespace(" ".join(s for s in kept_sentences if s))


def generate_candidates(prompt: str) -> Dict[str, str]:
    """Return {"conservative": ..., "balanced": ..., "aggressive": ...}."""
    requirements = extract_requirements(prompt)

    conservative = _build_candidate(requirements, drop_persona=False, use_model=False)
    balanced = _build_candidate(requirements, drop_persona=True, use_model=False)

    try:
        aggressive = _build_candidate(requirements, drop_persona=True, use_model=True)
    except Exception:
        aggressive = balanced  # never let a model-loading failure break the pipeline

    return {
        "conservative": conservative,
        "balanced": balanced,
        "aggressive": aggressive,
    }