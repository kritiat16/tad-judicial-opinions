"""
Generate TF-IDF, Legal-BERT, and GTE embeddings for judicial opinions.
Saves outputs to embeddings/.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

OPINIONS_PATH = Path(__file__).parent.parent / "data" / "processed" / "opinions_clean.json"
EMB_DIR = Path(__file__).parent.parent / "embeddings"

BATCH_SIZE = 8


# ── Mean pooling ──────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    """
    Mean pool token embeddings, ignoring padding positions.
    Padding tokens are zeroed out via the expanded attention mask so they
    contribute nothing to the sum; we divide by the number of real tokens.
    """
    token_embeddings = model_output.last_hidden_state          # (B, L, H)
    mask_expanded = (
        attention_mask.unsqueeze(-1)                           # (B, L, 1)
        .expand(token_embeddings.size())                       # (B, L, H)
        .float()
    )
    masked_sum = (token_embeddings * mask_expanded).sum(dim=1) # (B, H)
    token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)   # (B, H)
    return masked_sum / token_counts                           # (B, H)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_opinions():
    with open(OPINIONS_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    opinions, skipped = [], []
    for op in raw:
        text = op.get("clean_text", "").strip()
        if not text:
            skipped.append(op["id"])
            print(f"  [SKIP] {op['id']} — empty text")
        else:
            opinions.append(op)

    return opinions, skipped


# ── Model 1: TF-IDF ───────────────────────────────────────────────────────────

def run_tfidf(opinions):
    print("\n[1/3] TF-IDF")
    t0 = time.time()

    texts = [op["clean_text"] for op in opinions]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    print(f"  Matrix shape        : {matrix.shape}")
    print(f"  Vocabulary size     : {len(vectorizer.vocabulary_)}")

    scipy.sparse.save_npz(EMB_DIR / "tfidf_embeddings.npz", matrix)
    with open(EMB_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"  Saved tfidf_embeddings.npz and tfidf_vectorizer.pkl")
    print(f"  Time: {time.time() - t0:.1f}s")
    return matrix


# ── Model 2 & 3: transformer helper ──────────────────────────────────────────

def run_transformer_model(opinions, model_name, emb_filename, ids_filename):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device              : {device}")
    print(f"  Loading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    texts = [op["clean_text"] for op in opinions]
    ids = [op["id"] for op in opinions]

    all_embeddings = []
    processed_ids = []
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(texts))
            batch_texts = texts[start:end]
            batch_ids = ids[start:end]

            print(f"  Batch {batch_idx + 1}/{n_batches}", end="\r")

            try:
                encoded = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = model(**encoded)
                emb = mean_pooling(output, encoded["attention_mask"])
                all_embeddings.append(emb.cpu().numpy())
                processed_ids.extend(batch_ids)
            except Exception as batch_err:
                print(f"\n  [WARN] Batch {batch_idx + 1}/{n_batches} failed: {batch_err}")

    print()  # newline after \r progress line

    if not all_embeddings:
        raise RuntimeError(f"No batches succeeded for {model_name}")

    emb_array = np.vstack(all_embeddings)
    print(f"  Embedding shape     : {emb_array.shape}")

    np.save(EMB_DIR / emb_filename, emb_array)
    with open(EMB_DIR / ids_filename, "w", encoding="utf-8") as f:
        json.dump(processed_ids, f, indent=2)

    if len(processed_ids) < len(ids):
        print(f"  [WARN] Only {len(processed_ids)}/{len(ids)} opinions embedded")

    print(f"  Saved {emb_filename} and {ids_filename}")
    return emb_array, processed_ids


def run_legal_bert(opinions):
    print("\n[2/3] Legal-BERT  (nlpaueb/legal-bert-base-uncased)")
    t0 = time.time()
    result = run_transformer_model(
        opinions,
        model_name="nlpaueb/legal-bert-base-uncased",
        emb_filename="legal_bert_embeddings.npy",
        ids_filename="legal_bert_ids.json",
    )
    print(f"  Time: {time.time() - t0:.1f}s")
    return result


def run_gte(opinions):
    print("\n[3/3] GTE  (thenlper/gte-large)")
    t0 = time.time()
    result = run_transformer_model(
        opinions,
        model_name="thenlper/gte-large",
        emb_filename="gte_embeddings.npy",
        ids_filename="gte_ids.json",
    )
    print(f"  Time: {time.time() - t0:.1f}s")
    return result


# ── Cosine similarity matrices ────────────────────────────────────────────────

def compute_similarities(tfidf_matrix, legal_bert_emb, gte_emb):
    print("\n[Similarity matrices]")
    saved = []

    print("  TF-IDF ...", end=" ", flush=True)
    t0 = time.time()
    tfidf_sim = cosine_similarity(tfidf_matrix)
    np.save(EMB_DIR / "tfidf_similarity.npy", tfidf_sim)
    print(f"shape {tfidf_sim.shape}  ({time.time()-t0:.1f}s)")
    saved.append("tfidf_similarity.npy")

    if legal_bert_emb is not None:
        print("  Legal-BERT ...", end=" ", flush=True)
        t0 = time.time()
        lb_norm = normalize(legal_bert_emb, norm="l2")
        lb_sim = lb_norm @ lb_norm.T
        np.save(EMB_DIR / "legal_bert_similarity.npy", lb_sim)
        print(f"shape {lb_sim.shape}  ({time.time()-t0:.1f}s)")
        saved.append("legal_bert_similarity.npy")

    if gte_emb is not None:
        print("  GTE ...", end=" ", flush=True)
        t0 = time.time()
        gte_norm = normalize(gte_emb, norm="l2")
        gte_sim = gte_norm @ gte_norm.T
        np.save(EMB_DIR / "gte_similarity.npy", gte_sim)
        print(f"shape {gte_sim.shape}  ({time.time()-t0:.1f}s)")
        saved.append("gte_similarity.npy")

    return saved


# ── Opinion index ─────────────────────────────────────────────────────────────

def save_opinion_index(opinions):
    index = [
        {
            "position": i,
            "id": op["id"],
            "court": op["court"],
            "date_filed": op["date_filed"],
            "type": op["type"],
            "cluster_id": op["cluster_id"],
        }
        for i, op in enumerate(opinions)
    ]
    with open(EMB_DIR / "opinion_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"\n  Saved opinion_index.json  ({len(index)} entries)")
    return ["opinion_index.json"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Judicial Opinion Embeddings")
    print("=" * 60)

    print("\nLoading opinions...")
    opinions, skipped = load_opinions()
    print(f"  {len(opinions)} opinions to embed  |  {len(skipped)} skipped")

    saved_files = []

    # ── TF-IDF (always runs) ─────────────────────────────────────────────────
    tfidf_matrix = run_tfidf(opinions)
    saved_files += ["tfidf_embeddings.npz", "tfidf_vectorizer.pkl"]

    # ── Legal-BERT ────────────────────────────────────────────────────────────
    legal_bert_emb = None
    try:
        legal_bert_emb, _ = run_legal_bert(opinions)
        saved_files += ["legal_bert_embeddings.npy", "legal_bert_ids.json"]
    except Exception as e:
        print(f"\n  [ERROR] Legal-BERT failed: {e}")
        print("  Continuing with remaining models...")

    # ── GTE ───────────────────────────────────────────────────────────────────
    gte_emb = None
    try:
        gte_emb, _ = run_gte(opinions)
        saved_files += ["gte_embeddings.npy", "gte_ids.json"]
    except Exception as e:
        print(f"\n  [ERROR] GTE failed: {e}")
        print("  Continuing...")

    # ── Similarity matrices ───────────────────────────────────────────────────
    saved_files += compute_similarities(tfidf_matrix, legal_bert_emb, gte_emb)

    # ── Master index ──────────────────────────────────────────────────────────
    saved_files += save_opinion_index(opinions)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Opinions embedded : {len(opinions)}")
    if skipped:
        print(f"  Skipped           : {skipped}")
    print(f"  Files saved ({len(saved_files)}):")
    for name in saved_files:
        path = EMB_DIR / name
        size_kb = path.stat().st_size / 1024 if path.exists() else 0
        print(f"    {name:<40} {size_kb:7.1f} KB")


if __name__ == "__main__":
    main()
