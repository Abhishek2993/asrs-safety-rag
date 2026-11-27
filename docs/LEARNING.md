# How this project works, in plain language

This guide explains every idea and design decision in `asrs-safety-rag` so you can
explain it (and defend it) in an interview. Each section ends with **"Why this
choice?"** and, where useful, **"What would I say if asked..."**. Numbers quoted
here come from the runs reported in the README's *Evaluation results*.

---

## 1. The problem in one paragraph

NASA's Aviation Safety Reporting System (ASRS) holds hundreds of thousands of
voluntary, de-identified reports written by pilots, controllers, technicians and
cabin crew. A safety analyst reading a new report wants three things: *"have we
seen this before?"* (similar incidents), *"what contributed to it?"* (factor
classification) and *"what is the pattern across many reports?"* (trend summary).
The project answers all three with **retrieval** (find relevant reports), **machine
learning** (classify factors) and an optional **LLM** (write a summary) whose every
claim must be traceable to a report ID.

## 2. The data and how it is parsed

**The export format.** ASRS Database Online exports a CSV whose first *two* rows are
headers: row 1 is a *group* (`Time`, `Aircraft 1`, `Person 1`, `Events`,
`Assessments`, `Report 1`...) and row 2 the *field* (`Date`, `Flight Phase`,
`Narrative`...). The same field name occurs in several groups (`Aircraft 1 /
Flight Phase` and `Aircraft 2 / Flight Phase`), so you cannot just read row 2.
`load.py` finds the row containing `ACN`, takes the row above as groups (filling
blanks forward), and maps `(group, field)` pairs to clean column names.

**ACN** (accession number) is the unique report ID. Exports are capped at 10,000
records, so you download several files that may overlap; the loader **deduplicates
by ACN**, keeping the most complete copy.

**Multi-valued fields** such as *Contributing Factors / Situations* hold several values
separated by `;` and become Python lists.

*Why this choice?* Robust parsing is the unglamorous part that decides whether
everything downstream is trustworthy. The fixture-based tests cover blank group
cells, flattened headers, Windows-1252 encoding, embedded newlines and quotes,
footer rows and overlapping files.

## 3. Cleaning and the abbreviation dictionary

ASRS text mixes legacy shorthand (`ACFT`, `CLRNC`, `DSND`) with operational acronyms
(`TCAS`, `QRH`, `MEL`). `abbreviations.py` holds two small, documented tables:

* **shorthand** is *replaced* (`ACFT` -> `aircraft`) because the short form carries no
  extra meaning;
* **acronyms** are *kept* and expanded once (`TCAS (traffic collision avoidance
  system)`): an analyst searching for "TCAS" still gets a keyword match, and the
  embedding model sees the meaning.

Matching is case-sensitive on upper-case whole words, so "alt hold" or "Star" are
never touched. ASRS de-identification placeholders (`ZZZ`, `ZZZ1`, `XXR`) are
dropped from keyword search because they carry no information but are very
frequent.

**What goes into a document.** One document per report: a short metadata line
(flight phase, aircraft, operator), the synopsis and the narrative. Analyst-coded
fields (primary problem, contributing factors, anomaly) are **deliberately excluded**,
because they are the labels we evaluate against. Including them would be *label
leakage*: the system would look accurate only because the answer is in the input.

## 4. Embeddings

An **embedding** is a list of numbers (here 384) that represents the meaning of a
piece of text, produced by a neural network trained so that texts with similar
meaning get vectors pointing in similar directions. Similarity is measured with
**cosine similarity**: the cosine of the angle between two vectors (1 = same
direction, 0 = unrelated). Because all vectors are normalised to length 1, cosine
similarity is simply the dot product, which is what the FAISS index computes.

The model is **`all-MiniLM-L6-v2`** from sentence-transformers: 22M parameters, runs
on a laptop CPU, no API key, and a common baseline in retrieval papers.

**Chunking.** MiniLM reads at most 256 word-pieces (about 190 words) and silently
ignores the rest. ASRS narratives have a median of ~220 words and some exceed
1,500. So narratives are split into chunks of at most 160 words, aligned to
sentence boundaries, with ~40 words of overlap so an event described across a
boundary is still captured. A report's score is the score of its **best chunk**
(max-pooling): one highly relevant paragraph is enough to make a report relevant.

*What would I say if asked "why not a bigger model?"* Bigger embedding models
(e.g. bge-base, e5-large) usually retrieve better but are 5-15x slower on CPU. The
embedder is a parameter (`--embedder`), so swapping is one flag and the evaluation
harness tells you if it helped.

## 5. Keyword search: BM25

**BM25** scores a document by the query words it contains, weighting rare words
more (*inverse document frequency*: "windshear" is more informative than
"aircraft"), with diminishing returns for repeated words (parameter `k1`) and a
penalty for very long documents (parameter `b`). It is the classic search-engine
ranking function and remains a strong baseline.

`bm25.py` precomputes the BM25 weight of every (chunk, word) pair into one sparse
matrix, so scoring a query is a single sparse matrix-vector product - fast enough
to score all 28k chunks in a few milliseconds. A unit test checks it against a
hand computation.

## 6. Hybrid retrieval and Reciprocal Rank Fusion

Dense and keyword search fail differently:

* **dense** understands paraphrase ("tired" ~ "fatigued", "smell" ~ "fumes") but can
  miss exact identifiers and rare technical terms;
* **BM25** nails exact terms ("TCAS RA", "MEL", a waypoint) but misses paraphrase.

**Hybrid retrieval** runs both and merges the rankings with **Reciprocal Rank Fusion**
(RRF): each report gets `1/(60 + rank)` from each list, summed. RRF only uses
*ranks*, so there is no need to make cosine similarities (0-1) and BM25 scores
(0-50+) comparable, which is fragile. The constant 60 (from the original RRF paper)
dampens the advantage of being first so that agreement between the two lists
matters most.

**Metadata filters** (years, flight phase, aircraft, primary problem, factor label)
are applied *before* ranking: FAISS receives the list of allowed chunk IDs and BM25
scores of disallowed chunks are set to minus infinity. Filtering after ranking
would silently return fewer than *k* results.

*Engineering note.* The first version converted allowed report IDs into allowed
chunks with `np.isin` on string arrays, which took ~3 s per query on 28k chunks.
Mapping a boolean report mask through each chunk's report row takes ~1 ms; a query
now takes ~0.05 s. Profiling before optimising found this in one run.

## 7. Contributing-factor classification (multi-label)

**Multi-label** means each report can have *several* labels at once (e.g. human
factors + communication + weather), unlike multi-class where exactly one is right.
The standard approach is **one-vs-rest**: one yes/no classifier per label.

### The label taxonomy
ASRS codes about 20 contributing-factor values plus per-person human-factor codes.
`labels.py` maps them by *exact match* onto nine analysis labels (human factors,
fatigue, communication, weather, aircraft equipment, procedure, ATC/airspace,
organisational, airport/environment). Exact matching matters: "Environment - Non
Weather Related" contains the word *Weather* and must not count as weather.

### Approach A - supervised baseline
* **TF-IDF features**: each narrative becomes a sparse vector of word and two-word
  phrase weights (frequent-in-this-report but rare-overall terms weigh most).
* **Logistic regression** per label with `class_weight="balanced"`, which up-weights
  the rare positive class (fatigue appears in 2.6% of test reports).
* **Threshold tuning**: a probability of 0.5 is not automatically the right cut-off.
  For each label the threshold maximising F1 is chosen on **out-of-fold**
  predictions (5-fold cross-validation on the training set only), never on the test
  set - otherwise the test score would be optimistic.
* An **embedding** variant uses the mean MiniLM vector of the narrative's chunks as
  features instead of TF-IDF.

### Approach B - LLM with retrieved examples (RAG)
The prompt lists the label definitions, then (few-shot) the 4 most similar
*training* reports with their coded labels, then the new narrative, and asks for
JSON. Zero-shot is the same prompt without examples. The retrieval is restricted
to the training split so the test report (or its labels) can never leak into the
prompt.

### A third, cheap baseline - neighbour vote
Take the 10 most similar training reports; a label's score is the rank-weighted
share of neighbours carrying it. No training, no LLM; it shows how much the
retrieval alone knows.

*Why compare all three?* The LLM approach is the interesting one, but a safety
deliverable needs a **no-LLM baseline** that is deterministic, fast and auditable,
and a way to show whether the LLM actually adds value.

## 8. Evaluation metrics

### Classification
For one label, with TP = correctly predicted, FP = wrongly predicted, FN = missed:

* **precision** = TP / (TP + FP): of the reports we flagged, how many were right;
* **recall** = TP / (TP + FN): of the reports that truly had the label, how many we found;
* **F1** = harmonic mean of precision and recall (high only if both are high).

Across labels:

* **micro-F1** pools TP/FP/FN over all labels before computing F1, so frequent
  labels (human factors: ~60% of reports) dominate;
* **macro-F1** averages the per-label F1 scores, so every label counts equally and
  poor performance on rare labels (fatigue) is visible.

Report both: a high micro-F1 with a low macro-F1 means "good on common labels,
weak on rare ones".

**Confusion summary.** Multi-label tasks have no single confusion matrix, so the
evaluation counts *label swaps*: reports where label A was missed while label B was
wrongly predicted (e.g. *procedure* missed, *human factors* predicted instead).
This shows which categories the model cannot tell apart - often because the
categories genuinely overlap.

**Temporal split.** Besides a random 80/20 split, a model is trained on 2012-2019
and tested on 2020-2022. The drop measures **concept drift**: language and event
types change (COVID-19 staffing and masks appear only in later reports).

### Retrieval
* **precision@k**: share of the top *k* results that are relevant;
* **recall@k**: share of all relevant reports that appear in the top *k*;
* **MRR** (mean reciprocal rank): average of 1/rank of the first relevant result.

Two relevance definitions are used:

1. **Proxy (automatic, large):** a result is relevant if it has the same ASRS
   *primary problem* as the query report. Cheap and objective, but coarse
   ("Human Factors" covers thousands of very different events) and recall@k is
   tiny by construction (thousands of "relevant" reports). It is compared with
   **chance** (the precision a random ranking would get).
2. **Hand-labelled (small, precise):** 20 newly written queries with relevance
   judged against all 200 sample reports. Small but measures what users care about.

### Faithfulness of summaries
For each generated summary: **citation precision** = share of cited ACNs that are
in the evidence set given to the LLM; **statement coverage** = share of statements
with at least one valid citation. A lexical **support score** (share of a
statement's content words found in the cited reports) flags claims that cite a
real report but may not be supported by it.

## 9. Summaries and the citation check

Numbers (counts per year, top factors) are **computed with pandas**, never
generated: LLMs are unreliable at arithmetic and might invent figures. The LLM
only writes qualitative findings from a small **evidence set** of reports (selected
by retrieval or spread across the most frequent factors), and must end every
bullet with `[ACN ...]`.

`cite.py` then splits the text into statements, extracts citations, and checks
each against the evidence set. **Enforcement** removes any statement without a
valid citation before it is shown, and the removed statements are listed so a
reviewer can see what was dropped. Without an LLM, a template writes one cited
finding per top factor, so the requirement "every generated statement cites report
IDs" holds in every mode.

## 10. The LLM backend

`llm.py` defines one tiny interface (`generate(prompt, system, max_tokens,
json_mode)`) and several backends: a local **Ollama** server (default when
available), an in-process **transformers** model, any **OpenAI-compatible** API, and
the **Anthropic** API. Keys are read only from environment variables. `none` gives
a fully deterministic **no-LLM mode**.

*Why local first?* ASRS narratives are public, but many organisations cannot send
operational text to an external API. A small local model (Qwen2.5-1.5B-Instruct)
proves the pipeline works offline; a stronger model is one environment variable
away.

## 11. Design choices at a glance

| Choice | Alternative | Why |
|---|---|---|
| FAISS exact (`IndexFlatIP`) | approximate (HNSW/IVF) | 28k vectors search in ms exactly; approximation adds error for no gain at this size |
| BM25 implemented on a sparse matrix | `rank-bm25` package | ~100x faster scoring, 40 lines, unit-tested against hand computation |
| RRF fusion | weighted score sum | no score calibration needed; robust default |
| max-pool chunk scores | mean-pool | one relevant paragraph should make a report relevant |
| narrative-only classifier input | synopsis + narrative | a new report has no analyst synopsis; avoids leakage |
| per-label tuned thresholds | fixed 0.5 | rare labels need lower cut-offs to be found at all |
| stats computed, text generated | LLM writes everything | numbers must be exact and reproducible |
| citation enforcement | trust the LLM | hallucinated or wrong citations are the main hazard |

## 12. Questions you might get

* **"How do you know retrieval works?"** Two evaluations: an automatic proxy on 500
  queries (all methods about 2x chance; hybrid best at P@5, P@10 and MRR, dense best at
  P@1) and 20 hand-labelled queries, where hybrid is best on every metric (P@3 0.92 vs
  0.77 BM25 and 0.80 dense).
* **"Isn't 'same primary problem' a weak relevance signal?"** Yes - that is why it
  is compared against chance and complemented by hand labels; the README discusses
  its limits.
* **"What if the LLM invents a cause?"** Statements without a valid citation are
  removed; numbers are never generated; a lexical support score flags weakly
  supported statements; and the supervised baseline works without any LLM.
* **"Why is fatigue F1 low?"** ~3% prevalence and fatigue is often implicit ("long
  day", "red-eye"); see the per-label results and the class-imbalance hazard.
* **"Could this be used to blame a pilot?"** It must not: ASRS is voluntary and
  confidential by design; the tool supports pattern analysis only (see the ethics
  note).
