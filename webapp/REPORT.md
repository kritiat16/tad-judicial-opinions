# App Report: Topic, Not Stance

**Live app:** https://tad-judicial-opinions-s26.streamlit.app/  
**Authors:** Kriti Ajay & Feixiao Chen — NYU Text as Data, May 2026

---

## What the app does

This app accompanies the paper *Topic, Not Stance: A Validation 
Framework for Legal Opinion Similarity Using Circuit Splits*. It 
makes the paper's core finding interactive and accessible to a 
non-technical audience. The central finding — that NLP embedding 
models capture the topic of a legal opinion but not its legal 
stance — is difficult to convey in a static table. The app lets 
users see the validation results visually, understand the 
four-step pipeline, and test the similarity scorer on their own 
text.

The need it fills is practical: empirical legal scholars 
increasingly use NLP similarity methods to study judicial 
behavior, ideological polarization, and precedent diffusion. 
Before this project, there was no publicly available benchmark 
for evaluating whether those methods actually capture legal 
agreement or merely legal topic proximity. This app exposes 
the validation dataset, pipeline, and results in an accessible 
format and serves as a living demonstration of the methodology.

---

## The four sections

**1 — Overview**  
Introduces the project in plain English. Explains what a circuit 
split is, states the research question, and summarizes the key 
finding. Displays the four-step pipeline (Represent → Measure → 
Validate → Select) as styled visual boxes. Designed to be 
readable by anyone, including non-technical visitors.

**2 — Validation Results**  
Loads the full validation results from 
`embeddings/validation_results.json` and displays them as a 
styled table with seven columns: model, within-split mean, random 
baseline, gap, PR AUC, discrimination rate, and Mann-Whitney 
p-value. The Gap column is color-coded green (positive = not 
fooled by topic) and red (negative = fooled). A grouped bar chart 
shows within-split vs. random baseline means side by side for 
each model. Three plain-English takeaway boxes summarize each 
model's behavior. A conclusion box states the headline finding.

**3 — Similarity Explorer**  
An interactive demo where users paste two opinion excerpts and 
click "Compute TF-IDF Similarity" to get a live cosine similarity 
score. The score is displayed as a metric and a progress bar. 
A dynamic interpretation box explains what the score means — 
high similarity with opposite conclusions illustrates the topic 
confound the paper identifies. Pre-filled with real excerpts from 
split_001 (*Holmes v. Elephant Ins.* vs. *Baysal v. Midvale*) 
as a default example.

**4 — About**  
Project metadata: title, authors, course, links to the paper, 
GitHub repo, and validation dataset. Project details panel lists 
the data source, models evaluated, validation metrics, and ground 
truth sources.

---

## Technical implementation

The app is built with **Streamlit** and runs entirely from 
precomputed files — no model inference happens at runtime except 
for the live TF-IDF similarity computation.

**File loading:**  
On startup the app loads:
- `embeddings/validation_results.json` — validation metrics for 
  all three models, used to populate the results table and chart
- `embeddings/tfidf_vectorizer.pkl` — the fitted TF-IDF 
  vectorizer, used for live similarity computation

Both files are loaded with `st.cache_resource` to avoid 
reloading on every interaction.

**Live TF-IDF similarity:**  
When the user clicks "Compute TF-IDF Similarity", the app:
1. Reads the two text areas
2. Transforms both texts using the pre-fitted 
   `TfidfVectorizer` (same vocabulary as the validation run)
3. Computes cosine similarity between the two sparse vectors 
   using `sklearn.metrics.pairwise.cosine_similarity`
4. Displays the result as `st.metric` and `st.progress`

Legal-BERT and GTE are not run at inference time — they require 
GPU resources not available in the Streamlit Cloud free tier.

**Sidebar navigation:**  
Section switching is handled via `st.sidebar.radio`. Each section 
is a conditional block — only the selected section renders.

**Theming:**  
Pipeline step boxes use hardcoded dark blue backgrounds with 
explicit white text to ensure readability in both light and 
dark Streamlit themes.

---

## How to run locally

```bash
# Clone the repo
git clone https://github.com/kritiat16/tad-judicial-opinions.git
cd tad-judicial-opinions

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run webapp/app.py
```

The app will open at `http://localhost:8501`.

**Requirements:** The app needs 
`embeddings/tfidf_vectorizer.pkl` and 
`embeddings/validation_results.json` to be present. 
These are included in the repo. The large embedding 
files (`.npy`) are gitignored but are not needed to 
run the app.

---

## Live deployment

The app is deployed on Streamlit Community Cloud:  
**https://tad-judicial-opinions-s26.streamlit.app/**

Source: `webapp/app.py` on the `main` branch of  
`https://github.com/kritiat16/tad-judicial-opinions`
