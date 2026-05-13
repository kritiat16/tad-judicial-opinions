# Topic, Not Stance: A Validation Framework for Legal Opinion Similarity Using Circuit Splits

**Kriti Ajay & Feixiao Chen** — NYU Text as Data, May 2026

Do NLP embedding models capture legal *stance* or just legal *topic*? We evaluate three models against a hand-curated set of 30 federal circuit splits and find that none can distinguish opinions that legally disagree from randomly paired opinions.

---

## The four-step pipeline

| Step | What it does |
|------|-------------|
| 1. Represent | Map each opinion to a vector using TF-IDF, Legal-BERT, or GTE |
| 2. Measure | Compute pairwise cosine similarity across all opinion pairs |
| 3. Validate | Score each model against circuit split ground truth |
| 4. Select | Choose the model with strongest validation performance |

---

## Models evaluated

| Model | Type | Checkpoint |
|-------|------|-----------|
| TF-IDF | Bag-of-words baseline | sklearn TfidfVectorizer |
| Legal-BERT | Domain-specific transformer | nlpaueb/legal-bert-base-uncased |
| GTE | General semantic embeddings | thenlper/gte-base |

---

## Key results

| Model | Within-split A↔B | Random baseline | Gap | PR AUC | MWU p-value |
|-------|-----------------|-----------------|-----|--------|-------------|
| Legal-BERT | 0.913 | 0.913 | +0.0001 | 0.066 | 0.637 |
| GTE | 0.832 | 0.826 | -0.007 | 0.080 | 0.465 |
| TF-IDF | 0.125 | 0.086 | -0.039 | 0.151 | 0.556 |

Gap = random baseline − within-split mean.
Positive = not fooled by topic (desired).
MWU chance level = 0.017.

---

## Validation dataset

30 hand-curated federal circuit splits (2010–2025) across
10 doctrinal areas. Sources: Congressional Research Service
annual reports (R47899, R48369, R48846), Seton Hall Circuit
Review, ELSSCAP. Each split has named anchor cases on both
sides with CourtListener opinion IDs.

58 of 60 opinions successfully retrieved (96.7%).
28 evaluable splits after dropping 2 failed retrievals.

---

## How to run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Preprocess opinions:**
```bash
python notebooks/preprocess.py
```

**Generate embeddings:**
```bash
python notebooks/embed.py
```

**Run validation:**
```bash
python notebooks/validate.py
```

**Run the web app locally:**
```bash
streamlit run webapp/app.py
```

---

## Citation

If you use the validation dataset or pipeline, please cite:

> Ajay, K. & Chen, F. (2026). Topic, Not Stance: A Validation
> Framework for Legal Opinion Similarity Using Circuit Splits.
> NYU Text as Data course project.

This project extends:
> Ganguli, I., Lin, J., Meursault, V., & Reynolds, N. (2024).
> Patent Text and Long-Run Innovation Dynamics: The Critical
> Role of Model Selection. NBER Working Paper 32934.

---

## License

Data and code released for academic use.
Opinion texts sourced from CourtListener
(Free Law Project, CC BY 4.0).
