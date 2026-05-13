"""
webapp/app.py
Run from repo root: streamlit run webapp/app.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths (relative to repo root, where streamlit is invoked) ────────────────
REPO = Path(__file__).parent.parent
EMB_DIR = REPO / "embeddings"
RESULTS_PATH = EMB_DIR / "validation_results.json"
VECTORIZER_PATH = EMB_DIR / "tfidf_vectorizer.pkl"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Topic, Not Stance", layout="wide")

# ── Sidebar navigation ────────────────────────────────────────────────────────
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Validation Results", "Similarity Explorer", "About"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
if section == "Overview":
    st.title("Topic, Not Stance")
    st.subheader("Do NLP models capture legal stance or just legal topic?")

    st.markdown(
        """
        A **circuit split** occurs when two or more U.S. federal appellate courts reach
        opposite legal conclusions on the same question — for example, whether a hyperlink
        in an email constitutes sufficient notice for a filing deadline.
        We use 30 hand-curated circuit splits as ground truth to evaluate whether NLP
        embedding models can tell apart opinions that *disagree* from opinions that are
        merely on different topics.
        Following the four-step validation pipeline of Ganguli et al. (2024), we find
        that all three models tested — TF-IDF, Legal-BERT, and GTE — score legally
        opposing opinions just as similarly as randomly paired opinions, confirming that
        current embeddings capture legal *topic* but not legal *stance*.
        """
    )

    st.markdown("---")
    st.markdown("#### The four-step pipeline")

    cols = st.columns(4)
    steps = [
        ("Step 1 — Represent", "Map each opinion to a vector using TF-IDF, Legal-BERT, or GTE."),
        ("Step 2 — Measure", "Compute pairwise cosine similarity across all opinion pairs."),
        ("Step 3 — Validate", "Score each model against circuit-split ground truth."),
        ("Step 4 — Select", "Choose the model with strongest validation performance."),
    ]
    for col, (label, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div style="background-color: #2c5282; color: white; padding: 20px; '
                f'border-radius: 8px; text-align: center;">'
                f'<span style="color: white;"><b>{label}</b><br><br>{desc}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption(
        "Extending Ganguli et al. (2024) from patents to federal judicial opinions."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Validation Results
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Validation Results":
    st.title("Validation Results")

    if not RESULTS_PATH.exists():
        st.error(f"Results file not found: {RESULTS_PATH}\nRun `python notebooks/validate.py` first.")
        st.stop()

    with open(RESULTS_PATH, encoding="utf-8") as f:
        payload = json.load(f)

    models_raw = payload["models"]

    # Build display dataframe
    rows = []
    for m in models_raw:
        rows.append(
            {
                "Model": m["model"],
                "Within-split A↔B": m["within_split_ab_mean"],
                "Random baseline": m["random_baseline_mean"],
                "Gap": m["gap"],
                "PR AUC": m["pr_auc"],
                "MWU p-value": m["mwu_p"],
            }
        )
    df = pd.DataFrame(rows)

    def color_gap(val):
        color = "green" if val > 0 else "red"
        return f"color: {color}; font-weight: bold"

    styled = (
        df.style
        .map(color_gap, subset=["Gap"])
        .format(
            {
                "Within-split A↔B": "{:.4f}",
                "Random baseline": "{:.4f}",
                "Gap": "{:+.4f}",
                "PR AUC": "{:.4f}",
                "MWU p-value": "{:.3f}",
            }
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("#### Key takeaways")
    st.info("**Legal-BERT** (gap +0.0001, p=0.637): Not fooled by topic vocabulary — but also detects no stance signal.")
    st.info("**GTE** (gap -0.007, p=0.465): Slightly fooled by shared topic vocabulary.")
    st.info("**TF-IDF** (gap -0.039, p=0.556): Most fooled — bag-of-words scores inflate within-split similarity.")

    st.markdown("#### Within-split A↔B vs. Random baseline")
    import plotly.graph_objects as go

    fig = go.Figure(data=[
        go.Bar(name="Within-split A↔B",
               x=["Legal-BERT", "GTE", "TF-IDF"],
               y=[0.9133, 0.8324, 0.1254],
               marker_color="#1f4e79"),
        go.Bar(name="Random baseline",
               x=["Legal-BERT", "GTE", "TF-IDF"],
               y=[0.9134, 0.8256, 0.0863],
               marker_color="#7eb0d4"),
    ])
    fig.update_layout(
        barmode="group",
        title="Within-split vs Random Baseline by Model",
        yaxis_title="Mean cosine similarity",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Gap near zero confirms no stance detection")

    st.markdown("---")
    st.success(
        "All three models perform near chance (all MWU p > 0.45). "
        "Current embedding models capture legal topic, not legal stance."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Similarity Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Similarity Explorer":
    st.title("Similarity Explorer")
    st.subheader("See it yourself: paste two opinion excerpts")

    DEFAULT_A = (
        "The question before us is whether the posting of plaintiffs' driver's license "
        "numbers on the dark web constitutes a concrete injury sufficient to establish "
        "Article III standing. We hold that it does. The exposure of sensitive personal "
        "information to criminal actors on the dark web creates a material risk of "
        "identity theft that is both real and imminent."
    )
    DEFAULT_B = (
        "We conclude that the mere posting of license numbers on the dark web, without "
        "evidence of actual misuse or a substantially certain risk of imminent harm, does "
        "not establish the concrete injury required for Article III standing under Spokeo "
        "and TransUnion."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        text_a = st.text_area(
            "Opinion A — 4th Circuit, *Holmes v. Elephant Ins.*",
            value=DEFAULT_A,
            height=200,
        )
    with col_b:
        text_b = st.text_area(
            "Opinion B — 7th Circuit, *Baysal v. Midvale*",
            value=DEFAULT_B,
            height=200,
        )

    if st.button("Compute TF-IDF Similarity"):
        if not VECTORIZER_PATH.exists():
            st.error(
                f"Vectorizer not found: {VECTORIZER_PATH}\n"
                "Run `python notebooks/embed.py` first."
            )
        else:
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
            vecs = vectorizer.transform([text_a.strip(), text_b.strip()])
            score = float(cosine_similarity(vecs[0], vecs[1])[0, 0])

            st.metric(label="TF-IDF Cosine Similarity", value=f"{score:.4f}")
            st.progress(min(score, 1.0))

            if score >= 0.6:
                st.warning(
                    "⚠️ High similarity (score ≥ 0.6): These opinions share substantial "
                    "vocabulary and legal concepts. If they reach opposite conclusions on "
                    "the same question, this is exactly the topic confound this project "
                    "identifies — the model scores them as similar not because they agree, "
                    "but because they discuss the same doctrine using the same words."
                )
            elif score >= 0.3:
                st.info(
                    "↔️ Moderate similarity (score 0.3–0.6): These opinions share some "
                    "legal vocabulary but differ meaningfully in content. They may address "
                    "related but distinct legal questions, or discuss the same doctrine from "
                    "different angles. A score in this range does not tell you whether the "
                    "courts agree or disagree — only that they partially overlap in topic."
                )
            else:
                st.success(
                    "✓ Low similarity (score < 0.3): These opinions share little vocabulary "
                    "and are likely addressing different legal questions entirely. TF-IDF "
                    "correctly identifies them as topically unrelated. This is the range "
                    "where the model performs as expected — genuine topic distance produces "
                    "low similarity scores."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — About
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "About":
    st.title("About")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Topic, Not Stance")
        st.markdown(
            "A Validation Framework for Legal Opinion Similarity Using Circuit Splits"
        )
        st.markdown("**Authors:** Kriti Ajay & Feixiao Chen")
        st.markdown("**Course:** Text as Data, NYU, May 2026")
        st.markdown(
            "[Paper](https://github.com/kritiat16/tad-judicial-opinions/blob/main/paper/paper.md)"
            " &nbsp;|&nbsp; "
            "[GitHub](https://github.com/kritiat16/tad-judicial-opinions)"
            " &nbsp;|&nbsp; "
            "[Validation Dataset](https://github.com/kritiat16/tad-judicial-opinions/blob/main/data/validation/circuit_splits.csv)",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### Project details")
        st.markdown(
            """
            - **Data:** 30 hand-curated federal circuit splits (2010–2025)
            - **Models:** TF-IDF, Legal-BERT (`nlpaueb/legal-bert-base-uncased`),
              GTE (`thenlper/gte-base`)
            - **Validation:** Mann-Whitney U test, PR AUC, discrimination rate
            - **Ground truth source:** Congressional Research Service annual reports,
              Seton Hall Circuit Review
            """
        )
