"""
notebooks/validate.py

Evaluate embedding models against the circuit-split ground-truth pairs.

Methodology
-----------
Within-split A↔B pairs are **labeled negatives**: both opinions discuss the
same legal question but reach opposite conclusions.  A model that captures
legal meaning — rather than merely surface topic overlap — should score these
pairs NO HIGHER than random cross-split pairs.  The model whose within-split
A↔B mean is closest to (or below) the random baseline best distinguishes legal
agreement from topic similarity.

Within-side positives (A↔A or B↔B across different splits) are NOT used here.
Defining true positives would require knowing that Side A in one split legally
agrees with Side A in another split — information not encoded in the CSV.  The
`subject_area` grouping is too coarse: two Civil Rights cases may be on
opposite sides of their respective splits.  This limitation is noted in the
paper; future work could add an explicit cross-split alignment column.

Pipeline modes
--------------
Standalone mode (default):
    Computes embeddings inline from data/raw/validation_opinions.json.
    Runs immediately after fetch_courtlistener.py with no additional steps.

Full-pipeline mode:
    If the validation opinion IDs appear in embeddings/opinion_index.json
    (because preprocess.py + embed.py were re-run on the combined dataset),
    reads from the pre-built similarity matrices instead.

Pre-conditions (hard failures unless marked optional):
    embeddings/tfidf_similarity.npy          required
    embeddings/opinion_index.json            required
    data/raw/validation_opinions.json        required
    data/validation/circuit_splits.csv       required
    embeddings/legal_bert_similarity.npy     optional
    embeddings/gte_similarity.npy            optional
"""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import clean

REPO = Path(__file__).parent.parent
VAL_OPINIONS_PATH = REPO / "data" / "raw" / "validation_opinions.json"
SPLITS_CSV = REPO / "data" / "validation" / "circuit_splits.csv"
EMB_DIR = REPO / "embeddings"
RESULTS_PATH = EMB_DIR / "validation_results.json"

RANDOM_SEED = 42
N_RANDOM_SAMPLES = 1000


# ── Pre-condition checks ──────────────────────────────────────────────────────

def check_preconditions() -> None:
    required = [
        EMB_DIR / "tfidf_similarity.npy",
        EMB_DIR / "opinion_index.json",
        VAL_OPINIONS_PATH,
        SPLITS_CSV,
    ]
    optional = [
        EMB_DIR / "legal_bert_similarity.npy",
        EMB_DIR / "gte_similarity.npy",
    ]
    ok = True
    for p in required:
        if not p.exists():
            print(f"[ERROR] Required file missing: {p}")
            ok = False
    if not ok:
        sys.exit(1)
    for p in optional:
        if not p.exists():
            print(f"[WARN]  Optional file missing -- skipping that model: {p.name}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_validation_opinions() -> list[dict]:
    with open(VAL_OPINIONS_PATH, encoding="utf-8") as f:
        ops = json.load(f)
    for op in ops:
        op["clean_text"] = clean(op.get("plain_text", ""))
    valid = [op for op in ops if op["clean_text"].strip()]
    dropped = len(ops) - len(valid)
    if dropped:
        print(f"  [WARN] {dropped} opinion(s) had empty text after cleaning -- skipped")
    return valid


def load_splits() -> list[dict]:
    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_opinion_index() -> dict[str, int]:
    with open(EMB_DIR / "opinion_index.json", encoding="utf-8") as f:
        return {entry["id"]: entry["position"] for entry in json.load(f)}


# ── Inline embedding (standalone mode) ───────────────────────────────────────

def compute_tfidf_sim(opinions: list[dict]) -> tuple[np.ndarray, dict[str, int]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [op["clean_text"] for op in opinions]
    vec = TfidfVectorizer(stop_words="english", max_features=10000, sublinear_tf=True)
    sim = cosine_similarity(vec.fit_transform(texts))
    id_to_pos = {op["id"]: i for i, op in enumerate(opinions)}
    return sim, id_to_pos


def compute_transformer_sim(
    opinions: list[dict], model_name: str
) -> tuple[np.ndarray, dict[str, int]]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from sklearn.preprocessing import normalize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}  |  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    texts = [op["clean_text"] for op in opinions]
    BATCH_SIZE = 8
    all_emb = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch, truncation=True, max_length=512, padding=True, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            tok_emb = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
            pooled = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_emb.append(pooled.cpu().numpy())
            print(f"    Batch {i // BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}", end="\r")

    print()
    emb = normalize(np.vstack(all_emb), norm="l2")
    sim = emb @ emb.T
    id_to_pos = {op["id"]: i for i, op in enumerate(opinions)}
    return sim, id_to_pos


# ── Evaluation ────────────────────────────────────────────────────────────────

def _build_id(split_id: str, side: str) -> str:
    num = split_id.split("_", 1)[1] if "_" in split_id else split_id
    return f"op_split{num}_{side}"


def evaluate_model(
    model_name: str,
    sim_matrix: np.ndarray,
    id_to_pos: dict[str, int],
    splits: list[dict],
    all_val_ids: list[str],
) -> dict | None:
    """
    Compute:
      within_split_ab_mean  — mean A↔B cosine similarity within each split
                              (labeled negatives: same question, opposite holdings)
      random_baseline_mean  — mean similarity over N random cross-split pairs
      gap                   — random_baseline − within_split_ab_mean
                              positive = model pushes opposites below baseline (good)
      discrimination_rate   — fraction of splits where A↔B sim < median(A vs others)
                              higher = model better identifies legal disagreement
    """
    within_sims: list[tuple[str, str, str, float]] = []
    n_skipped = 0

    for split in splits:
        id_a = _build_id(split["split_id"], "A")
        id_b = _build_id(split["split_id"], "B")
        if id_a not in id_to_pos or id_b not in id_to_pos:
            n_skipped += 1
            continue
        pa, pb = id_to_pos[id_a], id_to_pos[id_b]
        within_sims.append((split["split_id"], id_a, id_b, float(sim_matrix[pa, pb])))

    if len(within_sims) < 3:
        print(
            f"  [SKIP] {model_name}: only {len(within_sims)} splits evaluable "
            "(need >= 3).  Were both sides fetched successfully?"
        )
        return None

    within_mean = float(np.mean([s for _, _, _, s in within_sims]))

    # Random baseline: sample pairs from all available validation opinion positions
    avail_positions = [id_to_pos[oid] for oid in all_val_ids if oid in id_to_pos]
    rng = random.Random(RANDOM_SEED)
    random_sims: list[float] = []
    if len(avail_positions) >= 2:
        for _ in range(N_RANDOM_SAMPLES):
            i, j = rng.sample(range(len(avail_positions)), 2)
            random_sims.append(float(sim_matrix[avail_positions[i], avail_positions[j]]))
    random_mean = float(np.mean(random_sims)) if random_sims else 0.0
    gap = random_mean - within_mean

    # Discrimination rate: for each split, is A↔B below the median of A vs all others?
    disc = 0
    for _, id_a, id_b, ab_sim in within_sims:
        pos_a = id_to_pos[id_a]
        other_pos = [
            id_to_pos[oid]
            for oid in all_val_ids
            if oid in id_to_pos and oid != id_a and oid != id_b
        ]
        if other_pos:
            a_vs_others = [float(sim_matrix[pos_a, p]) for p in other_pos]
            if ab_sim < float(np.median(a_vs_others)):
                disc += 1
    disc_rate = disc / len(within_sims)

    if gap > 0.05:
        interp = "GOOD  -- A<->B below baseline"
    elif gap > -0.05:
        interp = "NEUTRAL"
    else:
        interp = "BAD   -- A<->B above baseline"

    # PR AUC: label=1 for every evaluable within-split A↔B pair, 0 for all others.
    # Use ALL unique unordered pairs from available validation opinions so the
    # denominator matches the actual corpus (not a random sample).
    within_pair_set = {
        tuple(sorted([id_a, id_b])) for _, id_a, id_b, _ in within_sims
    }
    avail_ids = [oid for oid in all_val_ids if oid in id_to_pos]
    pr_labels: list[int] = []
    pr_scores: list[float] = []
    for ii in range(len(avail_ids)):
        for jj in range(ii + 1, len(avail_ids)):
            oid_a, oid_b = avail_ids[ii], avail_ids[jj]
            pair = tuple(sorted([oid_a, oid_b]))
            pr_labels.append(1 if pair in within_pair_set else 0)
            pr_scores.append(float(sim_matrix[id_to_pos[oid_a], id_to_pos[oid_b]]))

    pr_labels_arr = np.array(pr_labels)
    pr_scores_arr = np.array(pr_scores)
    n_positive = int(pr_labels_arr.sum())
    n_total = len(pr_labels_arr)
    chance_level = round(n_positive / n_total, 4) if n_total > 0 else 0.0
    pr_auc = (
        round(float(average_precision_score(pr_labels_arr, pr_scores_arr)), 4)
        if n_positive > 0 else 0.0
    )
    pr_auc_interp = "ABOVE CHANCE" if pr_auc >= 2 * chance_level else "AT CHANCE"

    # Mann-Whitney U test: within-split A<->B scores vs all cross-split scores
    within_scores_mwu = pr_scores_arr[pr_labels_arr == 1]
    cross_scores_mwu = pr_scores_arr[pr_labels_arr == 0]
    mwu_result = mannwhitneyu(within_scores_mwu, cross_scores_mwu, alternative="two-sided")
    mwu_u = round(float(mwu_result.statistic), 1)
    mwu_p = round(float(mwu_result.pvalue), 4)

    return {
        "model": model_name,
        "within_split_ab_mean": round(within_mean, 4),
        "random_baseline_mean": round(random_mean, 4),
        "gap": round(gap, 4),
        "discrimination_rate": round(disc_rate, 3),
        "pr_auc": pr_auc,
        "chance_level": chance_level,
        "pr_auc_interpretation": pr_auc_interp,
        "mwu_u": mwu_u,
        "mwu_p": mwu_p,
        "n_evaluated": len(within_sims),
        "n_skipped": n_skipped,
        "interpretation": interp,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("Circuit-Split Validation: Embedding Model Evaluation")
    print("=" * 68)

    check_preconditions()

    print("\nLoading data...")
    opinions = load_validation_opinions()
    splits = load_splits()
    existing_index = load_opinion_index()

    print(f"  Validation opinions loaded : {len(opinions)}")
    print(f"  Circuit splits in CSV      : {len(splits)}")
    print(f"  Existing opinion index     : {len(existing_index)} entries")

    # Decide mode: full-pipeline (validation IDs in index) vs standalone (compute inline)
    val_ids_in_index = {op["id"] for op in opinions if op["id"] in existing_index}
    use_existing = len(val_ids_in_index) >= 3

    if use_existing:
        print(f"\n  Mode: full-pipeline ({len(val_ids_in_index)} validation IDs found in index)")
    else:
        print(
            f"\n  Mode: standalone -- validation IDs not yet in opinion index "
            f"({len(val_ids_in_index)}/{len(opinions)} found).\n"
            f"  Computing embeddings inline from fetched text.\n"
            f"  Tip: re-run preprocess.py + embed.py on the combined dataset "
            f"to use pre-built matrices."
        )

    all_val_ids = [op["id"] for op in opinions]
    results: list[dict] = []

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    print("\n[1/3] TF-IDF")
    if use_existing:
        tfidf_mat = np.load(EMB_DIR / "tfidf_similarity.npy")
        id_to_pos: dict[str, int] = existing_index
    else:
        print("  Computing TF-IDF similarity inline...")
        tfidf_mat, id_to_pos = compute_tfidf_sim(opinions)
        print(f"  Matrix: {tfidf_mat.shape}")

    r = evaluate_model("TF-IDF", tfidf_mat, id_to_pos, splits, all_val_ids)
    if r:
        results.append(r)

    # ── Legal-BERT ────────────────────────────────────────────────────────────
    print("\n[2/3] Legal-BERT")
    if use_existing:
        if (EMB_DIR / "legal_bert_similarity.npy").exists():
            lb_mat = np.load(EMB_DIR / "legal_bert_similarity.npy")
            r = evaluate_model("Legal-BERT", lb_mat, existing_index, splits, all_val_ids)
            if r:
                results.append(r)
        else:
            print("  [SKIP] legal_bert_similarity.npy missing")
    else:
        try:
            print("  Computing Legal-BERT embeddings inline...")
            lb_mat, lb_pos = compute_transformer_sim(
                opinions, "nlpaueb/legal-bert-base-uncased"
            )
            r = evaluate_model("Legal-BERT", lb_mat, lb_pos, splits, all_val_ids)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  [SKIP] Legal-BERT failed: {exc}")

    # ── GTE ───────────────────────────────────────────────────────────────────
    print("\n[3/3] GTE")
    if use_existing:
        if (EMB_DIR / "gte_similarity.npy").exists():
            gte_mat = np.load(EMB_DIR / "gte_similarity.npy")
            r = evaluate_model("GTE", gte_mat, existing_index, splits, all_val_ids)
            if r:
                results.append(r)
        else:
            print("  [SKIP] gte_similarity.npy missing")
    else:
        try:
            print("  Computing GTE embeddings inline...")
            gte_mat, gte_pos = compute_transformer_sim(opinions, "thenlper/gte-base")
            r = evaluate_model("GTE", gte_mat, gte_pos, splits, all_val_ids)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  [SKIP] GTE failed: {exc}")

    if not results:
        print(
            "\n[ERROR] No models produced results.  "
            "Check that fetch_courtlistener.py ran successfully."
        )
        sys.exit(1)

    # Sort: best gap first
    results.sort(key=lambda x: x["gap"], reverse=True)

    # ── Output table ──────────────────────────────────────────────────────────
    W = 140
    print("\n" + "=" * W)
    print("Results -- ranked by gap  (random_baseline - within_split A<->B)")
    print("Positive gap = model pushes legally-opposing pairs below random baseline (better)")
    print(f"PR AUC chance level ~= n_positive / n_total  (AT CHANCE = within 2x chance)")
    print(f"MWU: Mann-Whitney U, two-sided  (p < 0.05 = within-split scores differ from cross-split)")
    print("=" * W)
    hdr = (
        f"{'Model':<13} {'Within A<->B':>11} {'Random base':>12} "
        f"{'Gap':>8} {'Disc rate':>10} {'PR AUC':>8} {'Chance':>8}  "
        f"{'PR interp':<14}  {'MWU U':>10} {'MWU p':>8}  {'Gap interp'}"
    )
    print(hdr)
    print("-" * W)
    for r in results:
        print(
            f"{r['model']:<13} {r['within_split_ab_mean']:>11.4f} "
            f"{r['random_baseline_mean']:>12.4f} {r['gap']:>8.4f} "
            f"{r['discrimination_rate']:>10.3f} {r['pr_auc']:>8.4f} "
            f"{r['chance_level']:>8.4f}  {r['pr_auc_interpretation']:<14}  "
            f"{r['mwu_u']:>10.1f} {r['mwu_p']:>8.4f}  "
            f"{r['interpretation']}"
        )
    print("=" * W)

    best = results[0]
    n_eval = best["n_evaluated"]
    print(f"\nEvaluated on {n_eval} split(s) with both sides successfully fetched.")
    print(
        "Within-side positives (A<->A across same-subject splits) not computed --\n"
        "no cross-split side-alignment data available in the current CSV."
    )

    # Save
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_splits_total": len(splits),
        "n_opinions_fetched": len(opinions),
        "evaluation_mode": "full_pipeline" if use_existing else "standalone_inline",
        "models": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
