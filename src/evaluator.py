"""
Runs the optimizer over data/test_prompts.csv, writes
experiments/results.csv, and prints honest aggregate benchmark stats.

Run with:  python -m src.evaluator   (from the project root)
"""

import csv
import statistics
from pathlib import Path

from .optimizer import optimize_prompt

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "test_prompts.csv"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "experiments" / "results.csv"


def run_benchmark():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        prompts = list(csv.DictReader(f))

    rows = []
    for row in prompts:
        prompt_text = row["prompt"]
        result = optimize_prompt(prompt_text)
        metrics = result["metrics"]

        # For rejected prompts, `result["metrics"]` is a fallback
        # original-vs-original placeholder (see optimizer.py) — it is NOT
        # what the best candidate actually achieved. Pull the real
        # best-attempted candidate's numbers from `strategies` instead, so
        # the CSV shows what was actually tried, not a placeholder.
        if result["accepted"]:
            best_attempt = metrics
            failure_detail = ""
        else:
            strategies = result.get("strategies", {})
            best_name = max(
                strategies, key=lambda n: strategies[n]["token_reduction_percent"]
            ) if strategies else None
            best_attempt = strategies.get(best_name, metrics) if best_name else metrics
            failure_detail = " | ".join(result.get("candidate_failures", []))

        rows.append({
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "accepted": result["accepted"],
            "strategy_used": result.get("strategy_used") or "",
            "original_tokens": best_attempt["original_tokens"],
            "optimized_tokens": best_attempt["optimized_tokens"],
            "tokens_saved": best_attempt["tokens_saved"],
            "token_reduction_percent": best_attempt["token_reduction_percent"],
            "semantic_similarity": best_attempt["semantic_similarity"],
            "requirement_score": result.get(
                "requirement_score",
                best_attempt.get("requirement_score", 1.0 if result["accepted"] else ""),
            ),
            "reason": result["reason"],
            "failure_detail": failure_detail,
        })

    RESULTS_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accepted_rows = [r for r in rows if r["accepted"]]
    all_reductions = [r["token_reduction_percent"] for r in rows]
    accepted_reductions = [r["token_reduction_percent"] for r in accepted_rows]
    all_similarities = [r["semantic_similarity"] for r in rows]
    req_scores = [r["requirement_score"] for r in rows if isinstance(r["requirement_score"], (int, float))]

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total prompts:                  {len(rows)}")
    print(f"Accepted:                       {len(accepted_rows)}")
    print(f"Acceptance rate:                {len(accepted_rows) / len(rows) * 100:.2f}%")
    print(f"Avg reduction (best attempt,")
    print(f"  incl. rejected candidates):   {statistics.mean(all_reductions):.2f}%")
    if accepted_reductions:
        print(f"Avg token reduction (accepted): {statistics.mean(accepted_reductions):.2f}%")
        print(f"Max token reduction:            {max(accepted_reductions):.2f}%")
    print(f"Avg semantic similarity:        {statistics.mean(all_similarities):.4f}")
    if req_scores:
        print(f"Avg requirement score:          {statistics.mean(req_scores):.4f}")
        print(f"Min requirement score:          {min(req_scores):.4f}")
    print("=" * 60)
    print(f"Full per-prompt results written to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_benchmark()