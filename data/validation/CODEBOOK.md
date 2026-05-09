# CODEBOOK — circuit_splits.csv

This CSV is the manually curated, labeled validation set of circuit splits used to evaluate which embedding model best captures legal semantic similarity between judicial opinions. Each row represents one documented circuit split, with named representative cases on both sides of the disagreement. It plays the same role in our evaluation that the patent-interference case pairs play in Ganguli et al. (2024): a ground-truth set of known semantic relationships used to score candidate models. The dataset covers 30 splits drawn from CRS annual reports, the Seton Hall Circuit Review, and related sources (see `README.md` for full sourcing detail).

---

## Column reference

| Column | Type | Description | Example value | Edge-case notes |
|---|---|---|---|---|
| `split_id` | string | Unique row identifier, zero-padded sequential integer prefixed with `split_`. | `split_001` | No gaps in the sequence for this 30-row set; append new rows with the next integer. |
| `subject_area` | categorical | Normalized doctrinal area used for topic-distribution analysis. Applied uniformly across all rows regardless of how the source document categorized the case. | `Civil Rights` | Source-label variants where they differed from this taxonomy are preserved in `notes`. |
| `split_topic` | string | Free-text description of the precise legal question on which the circuits disagree, phrased as a yes/no or A/B question. | `Whether EEOC hyperlink notice starts the 90-day right-to-sue period` | Phrasing is intentionally neutral (neither side's framing). May contain commas; always quoted in CSV. |
| `side_a_case` | string | Full case name for the Side A representative opinion, formatted as it appears in the reporter. | `Garcia-Gesualdo v. Honeywell Aerospace, Inc.` | May contain commas (always quoted). Where multiple circuits align on Side A, the most recent or most-cited case was chosen as representative; others are noted in `notes`. |
| `side_a_circuit` | categorical | Federal circuit that decided the Side A opinion. One of: First Circuit through Eleventh Circuit, D.C. Circuit, Federal Circuit. | `First Circuit` | Full name, not abbreviated (e.g., "Ninth Circuit", not "9th Cir."). |
| `side_a_year` | int | Year the Side A opinion was decided (four-digit). | `2025` | Year of the representative case only. Where Side A includes multiple circuits, the year reflects the chosen representative. |
| `side_a_citation` | string | Volume-reporter-page citation for the Side A opinion, in standard federal reporter format. | `135 F.4th 10` | Does not include the full parenthetical (court + year). A small number of rows were pending full citation at time of entry; these are flagged in `notes`. |
| `side_a_holding` | string | One-sentence summary of the Side A holding, written to be self-contained and parallel to `side_b_holding`. | `Hyperlink notice was inadequate without clear 90-day warning.` | Paraphrased, not a direct quote. Phrasing is intentionally parallel across `side_a_holding` and `side_b_holding` to enable human-readable comparison. |
| `side_b_case` | string | Full case name for the Side B representative opinion. | `McDonald v. St. Louis Univ.` | Same conventions as `side_a_case`. |
| `side_b_circuit` | categorical | Federal circuit that decided the Side B opinion. | `Eighth Circuit` | Same controlled vocabulary as `side_a_circuit`. |
| `side_b_year` | int | Year the Side B opinion was decided (four-digit). | `2024` | May be earlier than `side_a_year` — the "newer" side is not always Side A. |
| `side_b_citation` | string | Volume-reporter-page citation for the Side B opinion. | `109 F.4th 1068` | Same format as `side_a_citation`. |
| `side_b_holding` | string | One-sentence summary of the Side B holding, written to be parallel to `side_a_holding`. | `Hyperlink notice can adequately start the right-to-sue period.` | See notes on `side_a_holding`. |
| `courtlistener_a_url` | URL | Direct URL to the Side A opinion on CourtListener. | `https://www.courtlistener.com/opinion/...` | Use `side_a_citation` to look up any row where this is not yet populated. |
| `courtlistener_b_url` | URL | Direct URL to the Side B opinion on CourtListener. | `https://www.courtlistener.com/opinion/...` | Same as `courtlistener_a_url`. |
| `scotus_resolved` | string | Status flag indicating whether the Supreme Court has resolved this split. | `N / check later` | Not a clean boolean. Common values: `N / check later` (not resolved as of entry, but unverified), `Y` (resolved), `pending / check later` (cert granted or petition filed), `cert granted / check later`, `petition filed / check later`. All values with `/ check later` require verification against current SCOTUS dockets before treating the split as unresolved. |
| `source` | string | Bibliographic reference to the primary source where this split was identified. | `CRS 2025 report, Table 1, Civil Procedure entry, page TBD` | Free text. Multiple sources sometimes listed, separated by semicolons. `page TBD` entries were not yet cross-referenced to exact page numbers at time of entry. |
| `notes` | string | Free-text field for curation notes: circuit alignment context, selection rationale, caveat flags, and any source-label discrepancies. | `Good yes/no split on electronic EEOC notice adequacy.` | Often records which other circuits align on each side beyond the two representative cases. Important for interpreting `scotus_resolved` flags marked `check later`. |

---

## How this is used downstream

Each row defines one labeled pair: two opinions on opposite sides of a documented legal disagreement. The pipeline fetches both opinions from CourtListener using `side_a_citation` / `side_b_citation` (or `courtlistener_a_url` / `courtlistener_b_url` when populated), then embeds them with each candidate model — TF-IDF, Legal-BERT, GTE, and OpenAI `text-embedding-3-large`. The evaluation metric is whether within-split cross-side similarity (Side A vs. Side B from the *same* split) is lower than within-side similarity (Side A opinion compared to other Side A opinions on closely related legal questions). The model that best separates those two distributions — maximizing the gap between within-side and cross-side similarity — is the strongest candidate for the judicial-opinion similarity task.
