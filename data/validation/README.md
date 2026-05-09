# Validation set — circuit splits

## Purpose

This folder holds the manually curated circuit-split validation set for the judicial-opinion similarity project. Each row in `circuit_splits.csv` is one documented circuit split, with named representative cases on both sides of the legal disagreement. The set is used to score candidate embedding models (TF-IDF, Legal-BERT, GTE, OpenAI `text-embedding-3-large`) by measuring how well each model separates opinions that disagree legally from opinions that agree — mirroring the role of patent-interference case pairs in the Ganguli et al. (2024) NBER paper this project builds on.

---

## Sources

Three sources were used to identify and verify splits:

- **Congressional Research Service annual reports on circuit splits** — R47899 (2023), R48369 (2024), and R48846 (2025), available on [congress.gov](https://congress.gov). These reports catalogue every split that emerged or widened in a given year and remained unresolved as of the report date, organized by doctrinal area.

- **Seton Hall Circuit Review, "Current Circuit Splits" feature** — used for splits from earlier years (roughly 2010–2022) and as a cross-check against CRS entries. The Circuit Review's annual lists are more granular on older splits and cover years not yet reached by CRS reporting.

- **Supplementary sources for pre-2017 splits** — occasional reference to the Beim & Rader (2019) replication dataset and SCOTUSblog's "Petitions to Watch" archives, primarily used to verify resolution status and to surface representative case citations for splits that appeared in earlier Circuit Review volumes.

---

## Selection criteria

- Both sides of the split must have published, precedential opinions with clear majority reasoning. Per curiam decisions and memorandum dispositions were excluded.
- **Year filter: 2010 onward**, to ensure CourtListener coverage and modern citation conventions.
- **Excluded splits:** splits that were resolved by the Supreme Court immediately after emerging (before circuit opinions on both sides had time to develop independently), and splits where one side's reasoning was abrogated within the same year as the decision.
- Each split must be expressible as a clear **Side A vs. Side B** disagreement. Splits resolving on subtle methodological differences — such as a 3-factor vs. 4-factor test — were excluded in favor of splits with genuinely opposite outcomes on the same legal question.

---

## Distribution and balancing

The set was assembled with an explicit goal of diversity across year, doctrinal topic, and circuit — but structural features of the source material constrain how balanced any sample can be. See [Limitations](#limitations-of-the-distribution) below.

### Year distribution

Year counts below are by `side_a_year` (the representative "newer" case for each split).

| Year | Count |
|------|-------|
| 2025 | 6 |
| 2024 | 6 |
| 2023 | 6 |
| 2022 | 1 |
| 2021 | 1 |
| 2020 | 1 |
| 2017 | 3 |
| 2016 | 1 |
| 2014 | 1 |
| 2013 | 2 |
| 2012 | 1 |
| 2011 | 1 |
| **Total** | **30** |

### Topic distribution

Counts use the normalized `subject_area` column. Note that `split_019` carries a source typo (`Bankrupcy Law`) — counted here under `Bankruptcy`.

| Subject area | Count |
|---|---|
| Civil Rights | 5 |
| Criminal Law & Procedure | 5 |
| Immigration | 5 |
| Civil Procedure | 4 |
| Criminal Procedure | 3 |
| Labor & Employment | 3 |
| Bankruptcy | 2 |
| Arbitration | 1 |
| Environmental Law | 1 |
| Firearms | 1 |
| **Total** | **30** |

### Circuit distribution

Counts reflect the number of splits in which a circuit appears on either side (Side A or Side B). Circuits with multiple qualifying cases for the same split are counted once per split, not per case.

| Circuit | Appearances (either side) |
|---|---|
| Ninth Circuit | 11 |
| Seventh Circuit | 9 |
| Second Circuit | 8 |
| First Circuit | 7 |
| Fifth Circuit | 7 |
| Eighth Circuit | 4 |
| Fourth Circuit | 3 |
| Sixth Circuit | 3 |
| Tenth Circuit | 3 |
| Eleventh Circuit | 3 |
| Third Circuit | 2 |
| D.C. Circuit | 0 |

---

### Limitations of the distribution

We aimed for balance across year, topic, and circuit, but several structural constraints made perfect balance impossible:

- **Circuit imbalance by year.** Some circuits produced very few or no qualifying splits in certain years. We did not artificially pad these gaps because doing so would have forced inclusion of lower-quality splits that fail the selection criteria above.

- **Topic taxonomy drift.** The CRS reports and the Seton Hall Circuit Review use different subject-area taxonomies, and both have shifted over time. A split categorized as "Civil Rights" in one source may appear under "Criminal Procedure" or "Civil Procedure" in another, depending on which aspect of the holding is emphasized. We applied a single normalized taxonomy in the `subject_area` column, but the underlying source labels are preserved in the `notes` column where they differed.

- **Topic frequency changes over time.** Some topics are more represented in recent years than in earlier ones, reflecting actual changes in litigation patterns rather than selection bias. Most notably, **Firearms** splits appear frequently in 2023–2025 (post-*Bruen*) but were rarely catalogued as a distinct topic around 2010 — at that time, similar issues were typically grouped under "Criminal Procedure" or "Constitutional Law." This is a feature of the legal landscape, not of our sampling, but it does mean topic shares are not stable across the time range.

- **Imbalanced topic counts in source.** The CRS reports themselves are imbalanced — the 2025 report has 29 Criminal Law entries and 1 Antitrust entry. Our selection is shaped by what was available, weighted toward doctrinal diversity rather than proportional representation.

---

## File contents

| File | Description |
|---|---|
| `circuit_splits.csv` | 30-row labeled validation set; one row per circuit split, with representative cases on both sides. |
| `CODEBOOK.md` | Column-by-column documentation: data type, description, example value, and edge-case notes. |
| `README.md` | This file. Sourcing methodology, selection criteria, distribution tables, and limitations. |

---

## How to use

Pipeline code reads `circuit_splits.csv`, joins to opinion text fetched from CourtListener using the `side_a_citation` / `side_b_citation` columns (or `courtlistener_a_url` / `courtlistener_b_url` where populated), and uses the Side A / Side B structure as labeled positive and negative pairs for model evaluation. See `CODEBOOK.md` for column-level details, including notes on the `scotus_resolved` flag — rows marked `pending / check later` or `cert granted / check later` should be verified against current SCOTUS dockets before treating them as unresolved splits.
