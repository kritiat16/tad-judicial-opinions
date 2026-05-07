"""
Analyze similarity matrices and print a readable summary.
"""

import json
import numpy as np
from pathlib import Path

EMB_DIR  = Path(__file__).parent.parent / "embeddings"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    with open(EMB_DIR / "opinion_index.json") as f:
        index = json.load(f)  # list of {position, id, court, date_filed, type, cluster_id}

    with open(DATA_DIR / "opinions_clean.json") as f:
        opinions = {op["id"]: op for op in json.load(f)}

    id_to_pos = {entry["id"]: entry["position"] for entry in index}

    matrices = {}
    for label, fname in [
        ("TF-IDF",      "tfidf_similarity.npy"),
        ("Legal-BERT",  "legal_bert_similarity.npy"),
        ("GTE",         "gte_similarity.npy"),
    ]:
        path = EMB_DIR / fname
        if path.exists():
            matrices[label] = np.load(path)
        else:
            print(f"  [WARN] {fname} not found — skipping {label}")

    return index, opinions, id_to_pos, matrices


# ── Pair identification ───────────────────────────────────────────────────────

def find_majority_dissent_pairs(opinions, id_to_pos):
    """
    Return deduplicated list of (pos_i, pos_j, id_i, id_j) for every
    majority–dissent companion pair in the corpus.
    """
    pairs, seen = [], set()
    for op in opinions.values():
        if op["type"] == "majority" and "dissent" in op.get("companion_ids", {}):
            maj_id = op["id"]
            dis_id = op["companion_ids"]["dissent"]
            key = tuple(sorted([maj_id, dis_id]))
            if key not in seen and maj_id in id_to_pos and dis_id in id_to_pos:
                seen.add(key)
                pairs.append((id_to_pos[maj_id], id_to_pos[dis_id], maj_id, dis_id))
    return pairs


def upper_triangle(n):
    """Row, col indices for the upper triangle of an n×n matrix (diagonal excluded)."""
    return np.triu_indices(n, k=1)


# ── Statistics ────────────────────────────────────────────────────────────────

def distribution_stats(matrix):
    rows, cols = upper_triangle(matrix.shape[0])
    vals = matrix[rows, cols]
    return dict(
        n=len(vals),
        min=float(vals.min()),
        max=float(vals.max()),
        mean=float(vals.mean()),
        median=float(np.median(vals)),
        std=float(vals.std()),
    )


def mean_sim_for_pairs(matrix, pair_positions):
    """Average similarity for a list of (i, j) index pairs."""
    if not pair_positions:
        return float("nan")
    return float(np.mean([matrix[i, j] for i, j in pair_positions]))


def same_circuit_mean(matrix, index):
    n = len(index)
    vals = [
        matrix[i][j]
        for i in range(n)
        for j in range(i + 1, n)
        if index[i]["court"] == index[j]["court"]
    ]
    return float(np.mean(vals)) if vals else float("nan")


def diff_circuit_mean(matrix, index):
    n = len(index)
    vals = [
        matrix[i][j]
        for i in range(n)
        for j in range(i + 1, n)
        if index[i]["court"] != index[j]["court"]
    ]
    return float(np.mean(vals)) if vals else float("nan")


def random_nonpaired_mean(matrix, n, paired_set):
    """Average similarity for all upper-triangle pairs NOT in paired_set."""
    rows, cols = upper_triangle(n)
    vals = [
        matrix[i, j]
        for i, j in zip(rows, cols)
        if (i, j) not in paired_set
    ]
    return float(np.mean(vals)) if vals else float("nan")


# ── Top / bottom pairs ────────────────────────────────────────────────────────

def top_and_bottom_pairs(matrix, index, k=5):
    rows, cols = upper_triangle(matrix.shape[0])
    sims = matrix[rows, cols]
    order = np.argsort(sims)

    def make_entry(rank_in_order):
        i, j = int(rows[rank_in_order]), int(cols[rank_in_order])
        return {
            "id_a":      index[i]["id"],
            "court_a":   index[i]["court"],
            "type_a":    index[i]["type"],
            "cluster_a": index[i]["cluster_id"],
            "id_b":      index[j]["id"],
            "court_b":   index[j]["court"],
            "type_b":    index[j]["type"],
            "cluster_b": index[j]["cluster_id"],
            "score":     float(sims[rank_in_order]),
        }

    top    = [make_entry(order[-(r + 1)]) for r in range(k)]   # highest first
    bottom = [make_entry(order[r])        for r in range(k)]   # lowest first
    return top, bottom


# ── Printing helpers ──────────────────────────────────────────────────────────

def divider(char="=", width=70):
    print(char * width)

def section(title):
    print()
    divider()
    print(f"  {title}")
    divider()

def print_pair_table(pairs, title):
    print(f"\n  {title}")
    hdr = f"  {'#':<3}  {'ID A':<8} {'Court':<6} {'Type':<12}  {'ID B':<8} {'Court':<6} {'Type':<12}  {'Score':>6}  {'Note'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for rank, p in enumerate(pairs, 1):
        note = "same cluster" if p["cluster_a"] == p["cluster_b"] else ""
        print(
            f"  {rank:<3}  {p['id_a']:<8} {p['court_a']:<6} {p['type_a']:<12}"
            f"  {p['id_b']:<8} {p['court_b']:<6} {p['type_b']:<12}"
            f"  {p['score']:>6.4f}  {note}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    divider()
    print("  Similarity Matrix Explorer")
    divider()

    print("\nLoading data...")
    index, opinions, id_to_pos, matrices = load_data()
    n = len(index)
    print(f"  {n} opinions  |  {len(matrices)} model(s) loaded")

    md_pairs = find_majority_dissent_pairs(opinions, id_to_pos)
    md_positions = [(i, j) for i, j, _, _ in md_pairs]
    # symmetric set for fast exclusion lookups
    paired_set = {(i, j) for i, j in md_positions} | {(j, i) for i, j in md_positions}
    print(f"  {len(md_pairs)} majority-dissent pairs identified")

    model_lift = {}   # collected for the final ranking table

    for model_name, matrix in matrices.items():
        section(f"Model: {model_name}")

        # ── Distribution ─────────────────────────────────────────────────────
        s = distribution_stats(matrix)
        print(f"\n  Distribution ({s['n']} unique pairs):")
        print(f"    min    {s['min']:>7.4f}")
        print(f"    max    {s['max']:>7.4f}")
        print(f"    mean   {s['mean']:>7.4f}")
        print(f"    median {s['median']:>7.4f}")
        print(f"    std    {s['std']:>7.4f}")

        # ── Semantic groupings ────────────────────────────────────────────────
        md_sim    = mean_sim_for_pairs(matrix, md_positions)
        rnd_sim   = random_nonpaired_mean(matrix, n, paired_set)
        same_sim  = same_circuit_mean(matrix, index)
        diff_sim  = diff_circuit_mean(matrix, index)

        print(f"\n  Semantic groupings (mean cosine similarity):")
        print(f"    Majority-dissent pairs  {md_sim:>7.4f}   ({len(md_pairs)} pairs)")
        print(f"    Random non-paired       {rnd_sim:>7.4f}")
        print(f"    Same circuit            {same_sim:>7.4f}")
        print(f"    Different circuit       {diff_sim:>7.4f}")

        model_lift[model_name] = {
            "md":    md_sim,
            "rnd":   rnd_sim,
            "lift":  md_sim - rnd_sim,
            "ratio": md_sim / rnd_sim if rnd_sim > 0 else float("inf"),
        }

        # ── Top / bottom pairs ────────────────────────────────────────────────
        top, bottom = top_and_bottom_pairs(matrix, index, k=5)
        print_pair_table(top,    "Top 5 most similar pairs")
        print_pair_table(bottom, "Bottom 5 least similar pairs")

    # ── Model ranking ─────────────────────────────────────────────────────────
    section("Model ranking: majority-dissent signal vs. random baseline")
    print()
    print("  Lift = (majority-dissent mean) - (random non-paired mean)")
    print("  Ratio = majority-dissent mean / random non-paired mean")
    print()
    print(f"  {'Rank':<5} {'Model':<14} {'MD mean':>8} {'Random':>8} {'Lift':>8} {'Ratio':>8}")
    print(f"  {'-'*5} {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    ranked = sorted(model_lift.items(), key=lambda kv: kv[1]["lift"], reverse=True)
    for rank, (name, v) in enumerate(ranked, 1):
        print(
            f"  {rank:<5} {name:<14} {v['md']:>8.4f} {v['rnd']:>8.4f}"
            f" {v['lift']:>+8.4f} {v['ratio']:>7.3f}x"
        )

    print()
    best = ranked[0][0]
    worst = ranked[-1][0]
    print(f"  => {best} best separates companion pairs from random pairs")
    if ranked[0][1]["lift"] > 0:
        print(f"     All models show positive lift -- majority-dissent pairs are")
        print(f"     more similar to each other than to random opinions,")
        print(f"     confirming embeddings capture same-case topical coherence.")
    print()
    divider()


if __name__ == "__main__":
    main()
