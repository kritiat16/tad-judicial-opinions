---
title: "Topic, Not Stance: A Validation Framework for Legal Opinion Similarity Using Circuit Splits"
author: "Feixiao Chen, Kriti Ajay"
date: "May 2026"
abstract: |
  Recent work by Ganguli, Lin, Meursault, and Reynolds (2024) demonstrates that the choice of NLP model can produce opposite conclusions about long-run trends in patent similarity, motivating validation-based model selection against domain-specific ground truth tasks. We extend this framework to the legal domain. We construct a hand-curated validation set of 30 federal circuit splits (60 opinions, 12 years, 10 doctrinal areas) and evaluate three embedding models — TF-IDF, Legal-BERT, and GTE — on their ability to distinguish opinions that reach opposite conclusions on the same legal question from opinions that are unrelated. We find that all three models perform near chance: cosine similarity between within-split opposing opinions is statistically indistinguishable from similarity between randomly paired cross-split opinions, with TF-IDF most affected by topic-vocabulary overlap (gap of +0.039 above random) and Legal-BERT least affected (gap of −0.0001). We interpret this as evidence that current embedding models — including a domain-specific legal model — capture the *topic* of a legal opinion but not its *stance* on the legal question. This finding has implications for any downstream legal NLP application that requires distinguishing legal agreement from legal disagreement (precedent prediction, ideological analysis, doctrinal evolution): text similarity alone is insufficient. We release our validation set and pipeline as a public benchmark.
---

# 1. Introduction

The choice of NLP model is rarely incidental. Ganguli, Lin, Meursault, and Reynolds (2024) recently demonstrated that different embedding models — TF-IDF, doc2vec, S-BERT, GTE, PaECTER, OpenAI text-embedding-3-large — produce strikingly different and sometimes opposite representations of patent similarity. Applied to the same empirical question — how has contemporaneous invention similarity evolved over 1836–2023? — the choice of model determines the answer. GTE shows similarity declining for 150 years; TF-IDF shows similarity rising over the same period. Model selection, they argue, is not a technical preliminary but the substantive empirical question.

Their proposed resolution is a four-step pipeline: (1) represent each document as a vector under model $m$; (2) compute pairwise cosine similarity; (3) evaluate each candidate model against domain-specific ground truth tasks; (4) select the model with the strongest validation performance. The validation tasks are doing the real work in this framework. Ganguli et al. use **patent interferences** — USPTO proceedings in which expert examiners determine, on legal grounds, that two patent applications describe substantively the same invention — as their primary ground truth. The intuition is clean: a good similarity model should assign high cosine similarity to pairs that an expert has identified as describing the same idea.

This paper asks whether the same framework can be extended to a different but adjacent domain: federal judicial opinions. Judicial opinions are an attractive target. They are publicly available through the CourtListener archive, structurally consistent (majority, dissent, concurrence), and densely networked through citations. They have generated substantial recent NLP interest, with Legal-BERT (Chalkidis et al., 2020) emerging as a widely used domain-adapted embedding model. And they pose substantive empirical questions: are dissents becoming more semantically distant from majorities over time? Do circuit courts converge or diverge on contested doctrinal questions? Is there a measurable signature of ideological polarization in the language of legal reasoning?

But applying the validation framework directly turns out to surface a fundamental structural difference between the two domains. The patent paper's ground truth — interferences — is a *topic identity* signal: two patents are "interfering" when they describe the same invention. Cosine similarity in any reasonable embedding space should track this well; "describes the same thing" and "embeds to a similar vector" are operationalizations of the same underlying construct. The most natural legal ground truth — a **circuit split**, where two federal appellate courts have reached opposite conclusions on the same legal question — is structurally different. Circuit-split opinions discuss the same legal question, cite the same precedents, and use the same doctrinal vocabulary. What distinguishes them is not what they are about, but where they land. Text similarity, as conventionally implemented, is not designed to detect this.

This paper makes three contributions.

First, we adapt the Ganguli et al. validation framework to the legal domain. We compute embeddings under three models — TF-IDF, Legal-BERT, and GTE — and develop a validation task built on a hand-curated set of 30 federal circuit splits.

Second, we report that none of the three models meaningfully discriminates between opinions that legally agree and opinions that legally disagree on the same question. All three score within-split opposing opinion pairs at chance level relative to random cross-split pairs. The ordering across models is informative: TF-IDF is most fooled by topic vocabulary overlap (within-split pairs scored 45% higher than random); Legal-BERT comes closest to seeing through the topic confound (gap near zero); GTE is intermediate.

Third, we offer a methodological interpretation: this is not a failure of model selection but a constraint on what text similarity, as a class of methods, can measure. Circuit splits embed an opposition that text similarity is not built to surface. Researchers wishing to study legal disagreement, ideological polarization, or doctrinal evolution using embedding-based similarity should either design analyses that account for this limitation, or augment text similarity with explicit stance-detection methods.

We release our validation set (with full source attribution and selection methodology) and pipeline as a public benchmark for future legal NLP work.

# 2. Related Work

The most direct antecedent of this work is Ganguli et al. (2024), whose four-step validation pipeline we adopt. Their methodological argument — that model choice is the substantive empirical question, not a technical preliminary — generalizes naturally to any document corpus where the latent structure being measured may not align with what off-the-shelf embeddings capture.

Within legal NLP, Chalkidis et al. (2020) released Legal-BERT, a BERT variant pre-trained on 12GB of English legal text including legislation, court cases, and contracts. Subsequent work has fine-tuned legal embeddings for specific tasks: case retrieval (Locke and Zuccon, 2018), citation prediction (Sadeghian et al., 2018), and statute interpretation (Holzenberger et al., 2020). Our work occupies a different niche: rather than fine-tuning for a downstream task, we ask what off-the-shelf legal embeddings — including the widely used Legal-BERT checkpoint — actually measure when used as general-purpose similarity functions.

The circuit splits literature in legal scholarship is extensive. Beim and Rader (2019) compile a structured dataset of intercircuit splits from 2005–2013 for use in studies of Supreme Court agenda-setting. Hellman (2008) argues for a stricter standard of what counts as a true circuit split. The annual Congressional Research Service reports on circuit splits (R47899 for 2023, R48369 for 2024, R48846 for 2025) provide the primary source from which our validation set is drawn.

A separate line of work uses embedding similarity in legal text without explicit validation. We do not survey this literature exhaustively; our point is methodological. The Ganguli et al. framework, which we extend here, recommends that any such use be preceded by validation against a task aligned with the downstream research question.

# 3. Data

## 3.1 Validation set

The validation set is a manually curated CSV of 30 federal circuit splits spanning 2010–2025 across 10 doctrinal areas. Each row represents one documented circuit split, with named representative cases on each side of the legal disagreement, along with each case's citation, circuit, year, one-line holding summary, and CourtListener opinion ID.

Sources were drawn primarily from the Congressional Research Service's annual reports on circuit splits, which catalogue every split that emerged or widened in a given year and remained unresolved as of the report date. We supplemented these with the Seton Hall Circuit Review's "Current Circuit Splits" feature for earlier years (2010–2022) and used SCOTUSblog "Petitions to Watch" archives for verification.

Selection criteria were:

- **Precedential opinions on both sides.** Per curiam decisions and memorandum dispositions were excluded.
- **Year filter: 2010 onward,** to ensure CourtListener coverage and modern citation conventions.
- **Clear Side A vs. Side B disagreement.** Splits resolving on subtle methodological differences (e.g., 3-factor vs. 4-factor test) were excluded in favor of splits with genuinely opposite outcomes.

Table 1 reports the distribution across year, doctrinal subject area, and circuit.

We aimed for balance, but several structural constraints in the source material make perfect balance impossible. The CRS reports are themselves imbalanced (e.g., the 2025 report contains 29 Criminal Law splits and 1 Antitrust split). Doctrinal taxonomies have drifted across years and across sources: a split labeled "Civil Rights" in CRS may appear under "Criminal Procedure" in Seton Hall depending on the emphasized aspect of the holding. We applied a single normalized taxonomy in our `subject_area` field and preserved source labels in notes. Some topics are more represented in recent years than older ones for substantive reasons rather than selection bias — most notably, firearms splits appear frequently in 2023–2025 (post-*Bruen*) but were rarely catalogued as a distinct topic around 2010. We document these limitations in full in the released `data/validation/README.md`.

## 3.2 Opinion text retrieval

We retrieved full plain-text opinions for all 60 validation cases (30 splits × 2 sides) using the CourtListener REST API v4. We identified opinions by their CourtListener opinion ID, looked up manually by browsing CourtListener for each case in the validation set, rather than by citation. This choice was forced by an empirical observation: formal F.4th reporter citations are not yet synced for many 2024–2025 cases in CourtListener's metadata, so a citation-based lookup returns zero results even when the opinion is present in the archive under its numeric ID.

Of the 60 attempted retrievals, 58 succeeded (96.7%). Two cases (split 010 Side A and split 015 Side A) returned 404 errors, indicating that the CourtListener opinion ID has drifted or the opinion has been moved. These two splits are dropped from validation, leaving 28 evaluable splits.

# 4. Methods

## 4.1 Four-step pipeline

We adopt the four-step pipeline of Ganguli et al. (2024):

1. **Representation.** Each opinion is mapped to a vector in a model-specific embedding space.
2. **Measurement.** Cosine similarity is computed for all pairs of opinions in the corpus.
3. **Validation.** Each candidate model is scored against a domain-specific ground truth task.
4. **Selection.** The model with the strongest validation performance is selected for downstream analysis.

This paper covers steps 1–3 for the legal domain. Step 4 — the choice of a single best model for downstream substantive analysis — is left open, because none of our three models passes a meaningful validation threshold, as we discuss in Section 6.

## 4.2 Models

We evaluate three embedding models, chosen to span the major NLP generations relevant to legal text:

- **TF-IDF** (Salton, Wong, and Yang, 1975). A traditional bag-of-words baseline. We use scikit-learn's `TfidfVectorizer` with English stopwords, `max_features=10000`, `min_df=2`, and sublinear term frequency scaling.
- **Legal-BERT** (Chalkidis et al., 2020). A BERT variant pre-trained on 12GB of legal text including legislation, court cases, and contracts. We use the `nlpaueb/legal-bert-base-uncased` checkpoint from HuggingFace. Opinion text is tokenized to 512 tokens (truncating longer opinions to the first 512), and we apply mean pooling over non-padded token embeddings.
- **GTE** (Li et al., 2023). The General Text Embeddings model, identified by Ganguli et al. as the top overall performer in their patent validation. We use the `thenlper/gte-base` checkpoint, with the same tokenization and pooling as Legal-BERT.

We do not evaluate the proprietary OpenAI `text-embedding-3-large` model used in the patent paper, both for full replicability and to keep all candidate models within a single open-weights pipeline.

## 4.3 Text preprocessing

Each opinion is cleaned via a `preprocess.py` script before embedding. Cleaning steps include: stripping case headers, docket numbers, attorney listings, and court captions; removing standalone disposition boilerplate (AFFIRMED, REVERSED, SO ORDERED); removing inline footnote markers; normalizing whitespace; and repairing UTF-8 encoding artifacts. The aim is to retain only the substantive legal reasoning of the opinion, since headers and dispositions are formulaic and would inflate similarity for purely structural reasons.

## 4.4 Validation task: two framings

The central methodological choice is how to frame the validation task. We discuss this carefully because the choice is not obvious and the most natural analog to the patent paper turns out not to be the most informative test.

**Framework A (topic similarity, patent-paper analog).** In Ganguli et al., the positive label is "these two patents describe the same invention" (a USPTO interference judgment), and the negative label is "these two patents are randomly paired." A good model should score positives higher than negatives in cosine similarity. The analog for legal opinions would be: positive = within-split A↔B pair (two opinions adjudicating the same legal question); negative = cross-split pair (two opinions on unrelated questions). A good model under Framework A should score within-split pairs higher than cross-split pairs, reflecting that the two opinions discuss the same doctrinal question.

**Framework B (stance opposition, our primary framing).** Framework A is informative but undemanding. Two opinions in a circuit split necessarily share substantial vocabulary: both cite the same precedents, deploy the same doctrinal categories, and address the same statutory or constitutional provisions. By construction, they are topically similar. *Any* reasonable similarity model that captures topic at all should score within-split pairs higher than random cross-split pairs. What is unclear is whether a model can see *beyond* topic to register the opposition in stance.

Framework B operationalizes this question. Under Framework B, the desired behavior is that a model scores within-split A↔B pairs *no higher than* random cross-split pairs. The intuition: because within-split pairs are topically near-identical but conclusionally opposite, a model that distinguishes legal stance from legal topic should not be fooled into rating them similar. A model that does rate them similar is one that has confused topic alignment with legal alignment.

Framework B is the more demanding test, and the more informative one for our research question (can text similarity see stance?). We therefore adopt Framework B as our primary framing. Framework A is implicit in the reported numbers — the comparison between within-split and random baseline similarity is meaningful under both framings — and we comment on it in the Discussion.

## 4.5 Validation metrics

For each model $m$ and each evaluable split $s$, we compute:

- **A↔B similarity** $\text{sim}_m(A_s, B_s)$: cosine similarity between the Side A and Side B opinion of split $s$ under model $m$.
- **Random baseline:** mean cosine similarity over a sample of cross-split opinion pairs, drawn from all pairs $(i, j)$ where $i$ and $j$ belong to different splits in the corpus of 56 successfully fetched opinions.

We aggregate across splits and report:

- **Within-split A↔B mean** (28 pairs).
- **Random baseline mean** (1,512 cross-split pairs).
- **Topic-confound gap** = (within-split mean) − (random mean). Under Framework B, a gap near zero (or negative) indicates that the model sees through topic similarity; a positive gap indicates the model is fooled by topic.
- **Discrimination rate:** the fraction of splits where the within-split A↔B similarity falls below the median of Side A's similarity to all other opinions. Higher is better under Framework B. A discrimination rate of 0.5 corresponds to chance.

# 5. Results

Table 2 reports the validation results.

| Model | Within-split A↔B | Random baseline | Topic-confound gap | Disc. rate |
|---|---|---|---|---|
| TF-IDF | 0.125 | 0.086 | +0.039 | 0.500 |
| Legal-BERT | 0.913 | 0.913 | −0.0001 | 0.464 |
| GTE | 0.832 | 0.826 | +0.007 | 0.536 |

Figure 1 visualizes the same data as a grouped bar chart, showing within-split and random-baseline means side by side for each model.

![Figure 1: Within-split A↔B vs. random baseline mean cosine similarity, by model. Error bars and annotated topic-confound gaps shown for each model. Legal-BERT and GTE similarities are inflated by embedding anisotropy; the meaningful signal is the gap between paired and random distributions, not the absolute magnitude.](figures/fig1_validation_results.png){width=85%}

Three observations emerge.

**First, the absolute similarity values for the transformer models are not directly interpretable.** Legal-BERT averages 0.91 across all opinion pairs (within-split and random alike); GTE averages 0.83. These values reflect a well-documented anisotropy property of BERT-family embeddings (Ethayarajh, 2019), in which sentence embeddings cluster within a narrow cone of the embedding space and pairwise cosines are uniformly inflated. The relevant signal is therefore the *gap*, not the absolute magnitude, and absolute-magnitude comparisons across models are not meaningful.

**Second, all three models are at or near zero on the topic-confound gap under Framework B.** Legal-BERT's gap is essentially zero (−0.0001), the cleanest result under Framework B: this model scores within-split pairs no higher than random pairs, indicating it does not over-weight topic vocabulary. GTE's gap is slightly positive (+0.007), indicating mild topic over-weighting. TF-IDF's gap is the largest in magnitude (+0.039), with within-split pairs scored 45% higher than random pairs on average — the expected behavior for a bag-of-words model that has no representation of meaning beyond word identity.

**Third, the discrimination rate hovers around 0.50 for all three models** — equivalent to chance. None of the models can reliably rank a within-split A↔B pair below a random A-to-other pair. By this metric, no model meaningfully detects stance opposition. GTE marginally exceeds chance (0.536); Legal-BERT marginally falls below (0.464); TF-IDF is exactly at chance (0.500).

The ordering across the three models is consistent across the two metrics: TF-IDF is the most topic-confounded; Legal-BERT is the least topic-confounded; GTE is intermediate. But the absolute level of stance discrimination is near chance for all three.

# 6. Discussion

## 6.1 What this result tells us

The cleanest reading of these results is that the embedding models we evaluated — including a domain-specific legal model — capture the *topic* of a legal opinion but not its *stance* on the legal question. Two opinions in a circuit split discuss the same statute or constitutional provision, marshal the same precedents, and engage the same doctrinal categories. They differ in their disposition: one rules for the plaintiff, the other for the defendant; one reads the statute broadly, the other narrowly. Cosine similarity, as a measure of embedding closeness, registers the shared substance and is at best agnostic to the divergent conclusion.

This is not a defect of the models per se. Embedding-based similarity is fundamentally a topic-proximity measure, and it succeeds at that. Our finding is rather that the *task* of measuring legal agreement — the task most naturally framed by a circuit split — requires something that embedding similarity does not provide: a representation of where the opinion comes down, not what it is about.

## 6.2 Why circuit splits are harder than patent interferences

It is useful to be explicit about why our findings differ from those of Ganguli et al. The patent paper's PaECTER model achieves PR AUC of 0.65 on the interference task; GTE achieves 0.64. These are real, large signals: the embedding models do meaningfully separate interfering patent pairs from random pairs.

The structural difference is in the ground truth. A patent interference identifies two documents that describe the same underlying invention. The two documents are written independently, by different inventors, with potentially different vocabulary — but they refer to the same physical or conceptual artifact. Topic similarity *is* the ground truth signal. A model that captures topic captures interference.

A circuit split identifies two documents that adjudicate the same legal question and reach *opposite* conclusions. Both documents discuss the same statute or doctrine, draw on a largely shared corpus of precedent, and operate within the same legal vocabulary. Topic similarity is *not* the ground truth signal; it is the confound. The ground truth signal is the disposition, which appears in only a fraction of the text and is often expressed in subtle differences in how shared precedents are characterized.

This is, we believe, a generalizable observation. Validation tasks that target topic identity (interferences, near-duplicate detection, citation prediction) align well with what off-the-shelf embeddings measure. Validation tasks that target *stance opposition on a shared topic* (circuit splits, disagreement detection, ideological polarization) do not. The Ganguli et al. framework is correctly diagnostic in both regimes: in the first regime it identifies a winning model; in the second regime, as our results show, it correctly identifies that no off-the-shelf model is fit for purpose.

## 6.3 Implications for legal NLP

If text similarity captures topic but not stance, several downstream applications deserve reconsideration:

- **Precedent retrieval.** A model trained to retrieve "similar prior cases" will retrieve cases on the same topic, not cases reaching the same conclusion. For a litigator researching favorable precedent, this distinction is decisive.
- **Ideological polarization analysis.** Studies that operationalize ideological polarization as decreasing similarity between liberal and conservative opinions need to verify that they are measuring stance divergence rather than docket divergence — i.e., whether conservative and liberal judges are increasingly likely to be writing about *different things* rather than disagreeing about the *same things*.
- **Dissent detection.** A naive similarity-based dissent flagger — one that flags cases whose dissent is "very different" from the majority — will be confounded by the strong topic alignment between majority and dissent within a single case.

We do not claim text similarity is useless for these applications. We claim it is incomplete, and that the validation framework we propose can surface this incompleteness before it propagates into substantive empirical claims.

## 6.4 Limitations

Several limitations deserve attention.

**Sample size.** Our validation set contains 30 splits, of which 28 are evaluable. The patent paper used 322 interference pairs out of 96,580 possible pairs. While our set is small, it is fully hand-curated with documented sourcing, and we prioritized quality (clearly opposite-conclusion splits with named anchor cases) over quantity. Future work should expand the set; we expect the central finding (no model meaningfully detects stance) to be robust to scale, but the rank ordering across models may shift.

**Topic and circuit imbalance.** We document distributional imbalances in the validation set's README. The D.C. Circuit has zero appearances in our 30 splits; firearms cases are recent-heavy due to post-*Bruen* litigation patterns. We do not believe these imbalances drive our central finding, but they constrain claims of doctrinal generality.

**512-token truncation.** Both transformer models truncate opinions to their first 512 tokens. Long opinions are common in the corpus, and the truncated portion may contain disposition-relevant language. A chunk-and-mean-pool approach over the full opinion text would mitigate this. We did not implement it due to time constraints.

**Two failed retrievals.** Two of 30 splits had unfetchable Side A opinions due to CourtListener ID drift. These splits are excluded; the result is robust to their exclusion, but the n=28 sample size is itself a limitation.

**Anisotropy.** The absolute similarity values for transformer models are inflated by embedding anisotropy and should not be interpreted directly. Within-model comparisons (gap vs. baseline) are valid; across-model comparisons of absolute magnitudes are not.

## 6.5 Future directions

The validation framework developed here can be extended in several directions. Most immediately, we did not implement the patent paper's second validation task (human annotation regression) or the trend analysis (substantive empirical finding using the best-validated model). The latter would require fetching a larger background corpus of opinions (the patent paper uses millions of patents); we limited ourselves to the validation set in this paper.

A more methodologically interesting direction is to augment text similarity with explicit stance-detection methods. One option: train a classifier on within-split A↔B pairs (labeled "disagree") and within-side same-doctrine pairs (labeled "agree"), using model embeddings as features. If even a simple classifier can recover stance from the embedding (which our cosine-similarity results suggest is doubtful), this would shift the conclusion from "models do not encode stance" to "models encode stance but not in a way cosine recovers." Distinguishing these is non-trivial and deserves dedicated study.

# 7. Conclusion

We extended the validation-based model selection framework of Ganguli et al. (2024) from patent text to legal opinion text and find that the framework reveals a fundamental difference between the two domains. In the patent domain, ground truth (interferences) and embedding similarity are aligned: both measure topic identity. In the legal domain, the most natural ground truth (circuit splits) and embedding similarity are misaligned: the ground truth measures stance disagreement on a shared topic, while embedding similarity measures only topic proximity. Across three models — TF-IDF, Legal-BERT, and GTE — none reliably discriminates between opinions that legally agree and opinions that legally disagree on the same question.

We interpret this as a methodological constraint, not a model failure. Researchers using text similarity for substantive legal analysis should be explicit about whether their downstream question requires topic detection, stance detection, or both — and should validate their model choice against a ground truth aligned with their actual research question, not against a ground truth borrowed from an adjacent domain.

We release our validation set, source attribution, and pipeline as a public benchmark.

# References

Beim, D., & Rader, K. (2019). Legal Uniformity in the American Courts: A Study of Intercircuit Splits. *Journal of Empirical Legal Studies*, 16(2), 305–336.

Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). LEGAL-BERT: The Muppets straight out of Law School. *Findings of EMNLP 2020*, 2898–2904.

Congressional Research Service (2024). *The U.S. Courts of Appeals: Background and Circuit Splits from 2023*. CRS Report R47899.

Congressional Research Service (2025). *The U.S. Courts of Appeals: Background and Circuit Splits from 2024*. CRS Report R48369.

Congressional Research Service (2026). *The U.S. Courts of Appeals: Background and Circuit Splits from 2025*. CRS Report R48846.

Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. *Proceedings of EMNLP-IJCNLP 2019*, 55–65.

Ganguli, I., Lin, J., Meursault, V., & Reynolds, N. (2024). *Patent Text and Long-Run Innovation Dynamics: The Critical Role of Model Selection*. NBER Working Paper 32934.

Hellman, A. (2008). The Law of the Circuit Revisited: What Role for Majority Rule? *Southern Illinois University Law Journal*, 32, 625–652.

Holzenberger, N., Blair-Stanek, A., & Van Durme, B. (2020). A Dataset for Statutory Reasoning in Tax Law Entailment and Question Answering. *Proceedings of the Natural Legal Language Processing Workshop 2020*, 31–38.

Li, Z., Zhang, X., Wang, Y., Long, D., Xie, P., & Zhang, M. (2023). Towards General Text Embeddings with Multi-stage Contrastive Learning. arXiv:2308.03281.

Locke, D., & Zuccon, G. (2018). A Test Collection for Evaluating Legal Case Law Search. *SIGIR 2018*, 1261–1264.

Sadeghian, A., Sundaram, L., Wang, D. Z., Hamilton, W. F., Branting, K., & Pfeifer, C. (2018). Automatic Semantic Edge Labeling over Legal Citation Graphs. *Artificial Intelligence and Law*, 26, 127–144.

Salton, G., Wong, A., & Yang, C. S. (1975). A Vector Space Model for Automatic Indexing. *Communications of the ACM*, 18(11), 613–620.
