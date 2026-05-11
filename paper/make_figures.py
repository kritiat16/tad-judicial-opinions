"""
Generate Figure 1 for the paper: grouped bar chart of within-split A↔B vs.
random baseline mean cosine similarity for each of the three models.

Reads numbers from embeddings/validation_results.json if present; otherwise
falls back to the values reported in the validate.py run on 2026-05-11.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Hard-coded fallback values from the validate.py run on 2026-05-11.
FALLBACK = {
    "TF-IDF":     {"within_ab": 0.1252, "random": 0.0863},
    "Legal-BERT": {"within_ab": 0.9133, "random": 0.9134},
    "GTE":        {"within_ab": 0.8324, "random": 0.8256},
}

RESULTS_PATH = Path(__file__).parent.parent / "embeddings" / "validation_results.json"


def load_results() -> dict[str, dict[str, float]]:
    """Return {model_name: {within_ab, random}} from JSON or fallback constants."""
    if not RESULTS_PATH.exists():
        return FALLBACK
    with open(RESULTS_PATH) as f:
        payload = json.load(f)
    # validation_results.json stores a list under "models"; convert to lookup dict.
    out: dict[str, dict[str, float]] = {}
    for entry in payload.get("models", []):
        out[entry["model"]] = {
            "within_ab": entry["within_split_ab_mean"],
            "random":    entry["random_baseline_mean"],
        }
    return out if out else FALLBACK


def main() -> None:
    results = load_results()
    models = ["TF-IDF", "Legal-BERT", "GTE"]
    within   = [results[m]["within_ab"] for m in models]
    random_b = [results[m]["random"]    for m in models]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, within,   width,
           label="Within-split A↔B (n=28)",
           color="#4c72b0", edgecolor="white")
    ax.bar(x + width / 2, random_b, width,
           label="Random cross-split (n=1,512)",
           color="#dd8452", edgecolor="white")

    ax.set_ylabel("Mean cosine similarity", fontsize=11)
    ax.set_title("Within-split A↔B vs. random baseline similarity, by model",
                 fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.0)

    # Annotate gap above each pair; use 4 decimal places when gap rounds to ±0.000
    for i, m in enumerate(models):
        gap = within[i] - random_b[i]
        sign = "+" if gap >= 0 else ""
        decimals = 4 if abs(gap) < 0.0005 else 3
        ymax = max(within[i], random_b[i])
        ax.annotate(f"gap: {sign}{gap:.{decimals}f}",
                    xy=(i, ymax + 0.03),
                    ha="center", fontsize=9, color="#333")

    plt.tight_layout()
    out_path = OUT_DIR / "fig1_validation_results.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig1_validation_results.pdf", bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
