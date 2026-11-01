# API Token Minimizer

**Automated Prompt Engineering Tool to Minimize LLM Token Usage and Reduce
AI Inference Resource/Carbon Footprint**

A college mini-project (Deep Learning & AI) that takes a verbose LLM
prompt and produces a shorter version that uses fewer tokens while
provably preserving every requirement, constraint, number, format
instruction, and negative instruction in the original.

> **On the "carbon footprint" claim:** this project does **not** measure
> actual energy or carbon emissions. Every reported figure is an
> **estimated inference-resource saving derived from token reduction** —
> fewer tokens sent to an LLM API is a reasonable proxy for reduced
> compute cost, but it is not a calibrated energy/carbon measurement.
> The README, UI, and code all use this wording deliberately.

---

## 1. Problem & Motivation

LLM API costs and inference compute scale with the number of tokens
processed. Many real-world prompts contain persona framing, filler
phrases, and redundant wording that add tokens without adding
information. Naively shortening a prompt is dangerous, though — it's
very easy for an automatic compressor to silently drop something the
user actually needed (a required example count, an output format, a
"do not" instruction). This project builds a system that compresses
prompts **only when it can prove the compression is safe**.

## 2. Architecture

```
User Prompt
    │
    ▼
Tokenizer (tiktoken, gpt-4o-mini encoding)
    │
    ▼
Requirement / Constraint Extraction  (deterministic, regex-based)
    │  tags every sentence as one of:
    │  quantity | negative | format | inclusion | persona | general
    │  the first four are "critical" (hard requirements)
    ▼
Segment-Aware Hybrid Compressor
    │  - critical sentences: safe deterministic cleanup ONLY
    │    (filler-word removal + phrase simplification)
    │  - non-critical sentences (persona/filler): may be dropped, merged,
    │    or paraphrased by a transformer (FLAN-T5), since they carry no
    │    extracted requirement
    │  produces 3 candidates: conservative / balanced / aggressive
    ▼
Candidate Evaluation
    ├── Token count (tiktoken)
    ├── Whole-prompt semantic similarity (all-MiniLM-L6-v2 embeddings)
    ├── Per-requirement preservation (best-matching-sentence similarity
    │   + critical-term overlap + literal number/format checks)
    ▼
Safety Decision  (hard gates + soft score, see §5)
    │
    ├── at least one candidate passes  → return the one with the
    │                                     highest token reduction
    └── no candidate passes            → return the ORIGINAL prompt,
                                          unchanged, with reasons why
                                          every candidate failed
    ▼
Savings Report (tokens saved, % reduction, similarity, requirement
                score, estimated resource-savings note)
```

### Why this replaces the earlier "rewrite the whole prompt with FLAN-T5" design

The earlier version asked FLAN-T5 to paraphrase the *entire* prompt in
one shot. That gives the model free rein to drop anything, including
"use a comparison table" or "avoid unnecessary mathematical details" —
exactly the failure observed in testing (see §7). The fix is
architectural, not a bigger prompt or a lower threshold: **the model is
only ever allowed to touch sentences that don't carry a requirement.**
Every sentence tagged `critical=True` by the extractor is either kept
verbatim or passed through a small, hand-audited set of safe
substitutions (`src/text_utils.py::PHRASE_SIMPLIFICATIONS`) that cannot
delete a number, a format keyword, or a negation. This is a genuine
constrained-generation architecture (deterministic safety layer + a
real transformer model for the parts where generation is actually
safe), not "regex only."

## 3. AI / DL Components (for grading visibility)

- **Transformer sequence-to-sequence generation** — `google/flan-t5-base`
  paraphrases the non-critical segments in the "aggressive" strategy
  (`src/compressor.py::_flan_paraphrase`), using beam search
  (`num_beams=5, do_sample=False`) for deterministic, reproducible output.
  *(Random sampling was tried and rejected — see §8; it produced a
  1-token output with 0.03 similarity.)*
- **Sentence embeddings / semantic similarity** — `all-MiniLM-L6-v2`
  (`sentence-transformers`) embeds the original vs. optimized prompt and
  each requirement vs. each candidate sentence, using cosine similarity
  as the semantic-preservation signal (`src/metrics.py`, `src/validator.py`).
- **Constrained optimization** — the optimizer's objective is literally
  "maximize token reduction subject to similarity/requirement/format
  constraints" (`src/optimizer.py`), a textbook constrained-optimization
  framing over generated candidates.

## 4. Project Structure

```
API-Token-Minimizer/
├── app.py                        # Streamlit UI
├── requirements.txt
├── pytest.ini
├── README.md
├── src/
│   ├── __init__.py
│   ├── text_utils.py              # sentence splitting, safe filler/phrase rules
│   ├── tokenizer.py                # tiktoken counting (+ offline fallback)
│   ├── requirement_extractor.py    # regex-based requirement/constraint extraction
│   ├── compressor.py                # segment-aware hybrid compressor (deterministic + FLAN-T5)
│   ├── validator.py                 # per-sentence, hard/soft requirement validation
│   ├── metrics.py                   # token + semantic-similarity metrics
│   └── evaluator.py                 # benchmark runner
├── data/
│   └── test_prompts.csv            # 20-prompt benchmark (6 categories)
├── experiments/
│   └── results.csv                 # generated by evaluator.py
└── tests/
    ├── test_requirement_extractor.py
    ├── test_compressor.py
    ├── test_validator.py
    ├── test_optimizer.py
    └── test_metrics.py
```

## 5. Safety Checks (the actual acceptance criteria)

A candidate is accepted only if **all** of the following hold:

1. `optimized_tokens < original_tokens`
2. Whole-prompt semantic similarity (all-MiniLM-L6-v2) ≥ **0.85**
3. **All hard constraints preserved** — every `quantity`, `negative`, and
   `format` requirement individually survives (no averaging):
   - quantity: the exact number/word (e.g. "three") is still present,
     lexically, in the optimized prompt
   - format: the exact keyword (e.g. "table", "headings", "JSON") is
     still present — format instructions are not "safely paraphrasable,"
     so they require literal presence
   - negative: the negated sentence's best-matching optimized sentence
     has similarity ≥ 0.55 **and** at least 40% of its key terms survive
4. **Soft requirement score ≥ 0.85** — for `inclusion`/`general`
   requirements, each is checked against its *best-matching* optimized
   sentence (not the whole prompt — see §7 for why that distinction
   matters) using the same similarity + term-overlap test, then averaged.

If any check fails for every candidate, **the original prompt is
returned unchanged**, and the API/UI reports exactly which check each
candidate failed (see `candidate_failures` in the optimizer's return
value / the UI's "Why each candidate was rejected" panel). Thresholds
are fixed constants in `src/optimizer.py` and were **not** tuned down to
inflate acceptance — see §8 for why that would defeat the point of the
project.

## 6. Setup

```bash
cd API-Token-Minimizer
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

First run will download `google/flan-t5-base` and `all-MiniLM-L6-v2`
from Hugging Face (a few hundred MB total) — needs internet access once;
both are cached locally afterward.

## 7. Running It

### Single prompt (quick check)
```bash
python -c "from src.optimizer import optimize_prompt; import json; print(json.dumps(optimize_prompt('YOUR PROMPT HERE'), indent=2, default=str))"
```

### Streamlit UI
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`).
The UI lets you: enter a prompt, run optimization, see the optimized
prompt, compare token counts / reduction % / semantic similarity /
requirement score, see acceptance/rejection with a reason, see every
extracted requirement with its preservation status, compare all three
strategies side by side, and copy/download the optimized prompt.

### 20-prompt benchmark
```bash
python -m src.evaluator
```
This prints aggregate stats to the console and writes a full per-prompt
breakdown to `experiments/results.csv`.

### Tests
```bash
pytest -v
```
21 tests cover requirement extraction (quantities, negative constraints,
format constraints, persona detection), the compressor's deterministic
strategies, the validator's hard/soft gating logic, the optimizer's
fallback behavior, and token-reduction math. The validator and metrics
tests mock the embedding call (`src.validator.embed` /
`src.metrics.semantic_similarity`) so the suite runs without a model
download — this checks the *logic*, not the model; run the benchmark
(§ above) to exercise real embeddings end-to-end.

## 8. What Changed From the Earlier Version, and Why

The earlier implementation asked FLAN-T5 to rewrite the *entire* prompt
unrestricted, then validated by comparing one requirement's embedding
against the *whole* optimized prompt's embedding. On the 20-prompt
benchmark that produced:

| Metric | Earlier result |
|---|---|
| Acceptance rate | 10% (2/20) |
| Avg token reduction | 1.77% |
| Avg requirement score | 0.7454 |
| Min semantic similarity | 0.2306 |
| Min requirement score | 0.1750 |

Two root causes were identified and fixed architecturally, not patched:

1. **Unrestricted rewriting.** FLAN-T5 had no way to know "clear
   headings and a comparison table" was load-bearing — it's just text
   to shorten. *Fix:* the compressor now extracts requirements **first**
   and only lets the model touch sentences with no extracted
   requirement (`src/compressor.py`).
2. **Whole-prompt comparison.** Comparing one requirement's embedding to
   the embedding of an entire (possibly 60+ word) optimized prompt
   dilutes the signal in both directions — a requirement can look
   "kind of similar" to a long prompt that never actually mentions it,
   or a genuinely preserved requirement can be swamped by everything
   else in the prompt. *Fix:* the validator now compares each
   requirement against its single **best-matching sentence** in the
   candidate (`src/validator.py::_best_sentence_similarity`), and
   quantity/format/negative requirements get **hard, literal** checks
   instead of pure embedding similarity.

We also rejected random sampling (`do_sample=True`) for the FLAN-T5
step early on — it produced a 1-token output with 0.0331 similarity on
a test run — and kept deterministic beam search (`num_beams=5,
do_sample=False`) throughout.

## 9. Benchmark Results

**You must run `python -m src.evaluator` on your machine to get the
real, reportable numbers** — this repository does not ship pre-baked
results, and you should not report the numbers below as final without
re-running them yourself (model behavior/versions can shift results
slightly, which is expected and should be disclosed, not hidden).

What to look for when you run it:

| Metric | What it tells you |
|---|---|
| Acceptance rate | % of prompts where a safe compression was found at all |
| Avg token reduction (accepted) | how much the *safe* candidates actually saved |
| Max token reduction | best case observed |
| Avg semantic similarity | how close accepted candidates stayed to the original meaning |
| Avg / min requirement score | whether requirements are being preserved consistently, or only on easy prompts |

A credible report for a mentor demo states these numbers plainly,
alongside how many prompts were rejected and a specific example of a
rejection with its printed reason — that transparency is the point of
the safety-fallback design, not a weakness to hide.

## 10. Known Limitations

- **Sentence splitting is punctuation-based** (`. ! ?`) and can
  mis-split on abbreviations (e.g. "e.g.", "Dr.") in unusual prompts.
- **Requirement extraction is regex-based**, so a requirement phrased in
  a way that matches none of the patterns in
  `src/requirement_extractor.py` (no quantity words, no listed verb, no
  listed format keyword, no negation cue) will be classified as
  `general` and only soft-checked, not hard-gated.
- **Format-keyword checks are literal.** "Use a table" cannot be
  satisfied by an optimized prompt that says "structure it visually" —
  by design, since format instructions materially change what the
  downstream LLM outputs and are not safely paraphrasable.
- **The FLAN-T5 aggressive strategy only affects non-critical segments**,
  so on prompts that are almost entirely "critical" sentences (little to
  no persona/filler text), aggressive and balanced converge and the
  model contributes little extra compression — this is intentional
  (safety over aggressiveness) but worth stating plainly in a demo.
- **No fine-tuning.** FLAN-T5 and MiniLM are used off-the-shelf; a
  larger project could fine-tune on prompt-compression pairs.
- **Thresholds (0.85 similarity, 0.85 soft score, literal hard gates)
  are fixed constants**, not learned or calibrated against human
  judgments of "safe compression."
- **No true carbon/energy measurement** — see the note at the top of
  this file.

## 11. Future Work

- Fine-tune a small seq2seq model directly on (verbose prompt, safe
  compressed prompt) pairs instead of using FLAN-T5 zero-shot.
- Extend the extractor with a lightweight learned classifier (e.g. a
  small BERT-based sentence classifier) to catch requirements phrased
  outside the current regex patterns, while keeping the hard-gate logic.
- Add per-provider tokenizers (Claude, Gemini) alongside the OpenAI
  tiktoken counter.
- Add a real energy/carbon estimation methodology (e.g. hardware-level
  FLOP or joule estimates) if the project is extended beyond token
  counting, and only then upgrade the resource-savings claim.
