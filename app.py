import streamlit as st

from src.optimizer import optimize_prompt, SIMILARITY_THRESHOLD, SOFT_REQUIREMENT_THRESHOLD

st.set_page_config(page_title="API Token Minimizer", layout="wide")

st.title("API Token Minimizer")
st.caption(
    "Automated Prompt Engineering Tool to Minimize LLM Token Usage and "
    "Reduce AI Inference Resource Footprint"
)

with st.sidebar:
    st.header("About")
    st.write(
        "Compresses verbose prompts while preserving every extracted "
        "requirement, constraint, number, and output-format instruction. "
        "If no compression can be proven safe, the original prompt is "
        "returned unchanged — nothing is silently dropped."
    )
    st.markdown("---")
    st.write("**Safety thresholds**")
    st.write(f"- Whole-prompt semantic similarity ≥ {SIMILARITY_THRESHOLD}")
    st.write("- ALL hard constraints preserved (quantities, negatives, output formats)")
    st.write(f"- Soft requirement score (inclusion/general) ≥ {SOFT_REQUIREMENT_THRESHOLD}")

prompt = st.text_area(
    "Enter your prompt",
    height=200,
    placeholder="Paste a verbose prompt here...",
)

run_clicked = st.button("Optimize", type="primary")

if run_clicked:
    if not prompt.strip():
        st.error("Please enter a non-empty prompt.")
    else:
        try:
            with st.spinner("Extracting requirements and generating candidates..."):
                result = optimize_prompt(prompt)
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            st.stop()

        st.subheader("Result")
        if result["accepted"]:
            st.success(f"Accepted — strategy used: **{result['strategy_used']}**")
        else:
            st.warning("No safe candidate found — original prompt returned unchanged.")
        st.write(result["reason"])
        if not result["accepted"] and result.get("candidate_failures"):
            with st.expander("Why each candidate was rejected"):
                for line in result["candidate_failures"]:
                    st.write(f"- {line}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original prompt**")
            st.text_area("original_display", value=prompt, height=220, label_visibility="collapsed")
        with col_b:
            st.markdown("**Optimized prompt**")
            st.text_area(
                "optimized_display",
                value=result["optimized_prompt"],
                height=220,
                label_visibility="collapsed",
            )
            st.download_button(
                "Download optimized prompt",
                data=result["optimized_prompt"],
                file_name="optimized_prompt.txt",
            )

        m = result["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original tokens", m["original_tokens"])
        c2.metric("Optimized tokens", m["optimized_tokens"], delta=-m["tokens_saved"])
        c3.metric("Token reduction", f"{m['token_reduction_percent']}%")
        c4.metric("Semantic similarity", m["semantic_similarity"])
        st.caption(m["resource_savings_note"])

        if "requirement_score" in result:
            st.metric("Requirement preservation score", result["requirement_score"])

        st.markdown("---")
        st.subheader("Extracted requirements & preservation status")
        reqs = result.get("requirements")
        if reqs:
            for r in reqs:
                icon = "✅" if r["preserved"] else "❌"
                st.write(f"{icon} **[{r['kind']}]** {r['text']}  _(similarity: {r['similarity']})_")
        else:
            st.write("No hard requirements were extracted from this prompt.")

        st.markdown("---")
        st.subheader("Compare all strategies")
        for name, ev in result.get("strategies", {}).items():
            status = "PASSED" if ev["passes_safety"] else "rejected"
            with st.expander(f"{name} — {status} ({ev['token_reduction_percent']}% reduction)"):
                st.write(ev["candidate"])
                st.json({
                    "token_reduction_percent": ev["token_reduction_percent"],
                    "semantic_similarity": ev["semantic_similarity"],
                    "requirement_score": ev["requirement_score"],
                    "all_hard_constraints_preserved": ev["all_hard_constraints_preserved"],
                    "soft_score": ev["soft_score"],
                })