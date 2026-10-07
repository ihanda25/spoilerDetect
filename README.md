# Local spoiler detection with MiniLM

## Current direction — October 5, 2026

Target surfaces: headlines, comments, full reviews, and YouTube transcripts.
Future fine-tuning should use a **hosted GPU**; do not launch another local
training run without explicit approval. Prior local runs used Apple GPU/MPS.

Validation threshold tuning selected review-only epoch 2 (cutoff 0.215792).
Held-out IMDb test: precision 53.3%, recall 68.5%, F1 59.9%; see
`reports/selected-test/RESULTS.md`. Plot context did not improve the measured
validation comparison at matched recall; the earlier ceiling/generalization
claims below are historical interpretations, not established conclusions.

Next direction is data/error analysis, not an automatic extra epoch. The
TV Tropes sentence dataset audit is in `reports/tvtropes-audit/AUDIT.md`:
useful research candidate, with annotation/domain gaps and noncommercial
source terms requiring clarification for production use. No new training started.

Existing-model research is complete: `reports/model-research/comparison.md` and
`reports/model-research/replication-recipes.md`. RoBERTa's advertised 77.3% F1
from bhavyagiri is weighted F1, not spoiler-class F1.

A Colab T4 inference run tested Zritze RoBERTa on 64 synthetic diagnostics
(52 provisionally labeled, 12 uncertain). At cutoff .5, truncation caught 15/20
proposed spoilers with eight false positives; 512-token windows with 128-token
overlap caught 19/20 with the same eight false positives. All four late-reveal
cases were recovered. This is not human-reviewed production accuracy and no
threshold was tuned on these cases. See `reports/colab-roberta/RESULTS.md` and
`notebooks/README.md`. No new training occurred.

First annotation pass complete: 245 real-text candidates and a development-only threshold
selection pipeline with a held-out test gate. Resume instructions are in
`reports/product-evaluation/NEXT_RUN.md`; open `tools/review_spoiler_examples.html`
to label candidates without model predictions. The real-content Colab notebook
is `notebooks/Spoiler_Real_Content_Evaluation.ipynb`. Evaluation policy and evidence requirements:
`reports/product-evaluation/PLAN.md`. 31 human decisions are backed up separately; a gpt-6-luna agent completed
214 AI annotations at the user’s request. AI labels are kept distinct from human
review and cannot pass the current human evaluation gate. Human verification
is still required before reporting product benchmark metrics.

Three-model text-only development comparison completed on Colab T4: owned
MiniLM epoch2, Zritze RoBERTa, and Qwen2.5-3B-Instruct without RAG. No training or
API key. Human and AI annotation agreement stay separate; 98 held-out test rows
remain excluded. User authorized existing Colab units for the resumed LLM run; all three results
are saved in `reports/three-model-comparison/results.json`. Qwen abstained on
106/147 excerpts; human annotation policy needs clarification. No new purchases/API calls. Run status and notebook link: `reports/three-model-comparison/README.md`.

## Project history and current state (read this first)

October 5: the user authorized evaluating fixed RoBERTa on the same large IMDb validation/test splits as MiniLM. October 6: evaluation completed and downloaded predictions were locally verified. Validation-selected IMDb test spoiler F1: RoBERTa 58.4% vs MiniLM 59.9%. See `reports/roberta-imdb-test/RESULTS.md`; external model training overlap is unknown. See `reports/roberta-imdb-test/PLAN.md` and `STATUS.json`. No fine-tuning or purchases.

Current experiment: context-assisted comparison against fixed RoBERTa completed on the existing Colab T4 (October 5). See `reports/context-comparison/README.md` for frozen inputs, resume paths and status.
The user authorized AI judgments. Expanded AI development evaluation v2
is ready: 207 examples, 45 spoiler /158 safe /4 uncertain, with refreshed real
content and a separate source-checked plot challenge across 20 additional works.
See `reports/eval-refresh/EXPANDED_SET.md`. V2 results are in `reports/context-comparison/RESULTS.md`; this is an AI
diagnostic pack, not human gold. Original labels and held-out test are preserved;
no further training has started.

This project has gone through several experiments, in order. Each one's plan/
status file is the durable source of truth for that experiment — written so a
fresh session with no memory of prior conversations can resume correctly. If
anything below conflicts with one of those files, trust the file.

1. **Pilot** — `reports/FINETUNING_RESULTS.md`. MiniLM-L12-H384, 8,000 train
   reviews, 192-token truncation. F1=0.509. Complete, small-scale proof of concept.
2. **v1, abandoned** — `reports/full/`. Same model, all 444,644 train reviews,
   full token coverage via overlapping 512-token windows + max-pooling. Killed
   before any checkpoint saved: measured ETA was ~17 days/epoch on this Mac's
   MPS. A follow-up try with ModernBERT-base (native 8192-token context, to
   avoid windowing) OOM'd during calibration on this machine's 24GB of unified
   memory — also abandoned, weights deleted.
3. **v2, done through epoch 2, paused for a decision** — `reports/full-v2/`.
   Same MiniLM, all 444,644 reviews, but plain 512-token truncation instead of
   windowing (no overlapping windows, no max-pooling). Real measured cost
   ~4.6-5.4h/epoch. Epoch 1: F1=0.560 (best checkpoint, kept). Epoch 2:
   F1=0.534 (regressed at the default threshold, though ranking metrics
   roc_auc/average_precision both improved). The user chose not to run epoch 3,
   opting instead to try v3 below. Later validation threshold tuning favored
   epoch 2; these runs do not establish a model-performance ceiling.
4. **v3, done through epoch 1, awaiting a decision on epoch 2** —
   `reports/full-v2-plot/`. **Read `reports/full-v2-plot/PLAN.md` first.** Same
   MiniLM, same 444,644 reviews, same "plain truncation, no windowing"
   approach, but each input is now a pair — `[CLS] plot_summary [SEP] review
   [SEP]` — instead of review text alone, so the model can compare the review
   against what's actually known to happen in that movie (direct lookup by
   `movie_id`, not corpus retrieval). Same movie-disjoint splits as prior experiments. The plot version also
   shortens review text, so it does not isolate the effect of plot context. Epoch 1: F1=0.556 (~flat vs. v2's 0.560), but **recall improved to
   0.540** (up from v2's 0.487 — the specific weakness this experiment targets),
   at the cost of precision (0.573 vs 0.659) and ranking quality (roc_auc=0.788,
   AP=0.599, both down from v2). Mixed result: better at catching spoilers,
   worse at ranking confidence. No auto-continuation — the user decides on
   epoch 2 after seeing each result.

   **Run exactly one epoch per invocation, then stop — no auto-continuation for
   this experiment; the user reviews every epoch's result before the next one
   runs:**
   ```sh
   .venv/bin/python scripts/full_finetune_v2_plot.py
   ```
   Resumes automatically from the last saved epoch. Progress:
   `reports/full-v2-plot/status.json`, `reports/full-v2-plot/training.log`.
   Final eval (only once the user decides to stop):
   `.venv/bin/python scripts/full_finetune_v2_plot_final_eval.py` →
   `reports/full-v2-plot/FULL_V2_PLOT_RESULTS.md` (includes a side-by-side vs.
   v2's review-only numbers).

Operational lesson learned the hard way in v2 and worth repeating for any future
run: **this Mac must stay awake for training to progress** — attach
`caffeinate -dimsu -w <training_pid>` (detached from the shell) right after
launching, not after noticing a stall.

Collected the original **IMDb Spoiler Dataset**, version 1, by **Rishabh Misra**.
The raw dataset has not been modified. A local MiniLM fine-tuning experiment uses
movie-disjoint samples derived from this data.

## Fine-tuning experiment

The before/after report is generated at `reports/FINETUNING_RESULTS.md` when training
finishes. It compares an untrained classification head on pretrained MiniLM with the
fine-tuned checkpoint on the same test set. The untrained head is not a zero-shot
spoiler classifier. Scores are not calibrated probabilities.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-training.txt
python3 scripts/download_minilm.py
python3 scripts/prepare_training.py
.venv/bin/python scripts/finetune.py
.venv/bin/python scripts/predict.py "The acting was excellent."
```

Training uses 8,000 reviews, validation 1,000, and test 1,000. It updates all model
weights over two epochs, choosing the checkpoint by validation F1. It runs on Apple
MPS when available, otherwise CPU. A fixed 192-token limit keeps this initial local
experiment manageable; longer reviews are truncated, which limits detection.
After downloading dependencies and the base checkpoint, training and prediction
run offline. The pilot weights were subsequently deleted; its reports remain.
The current selected model is epoch 2 in `models/minilm-full/checkpoint.pt`;
`models/minilm-full/best/` is the older epoch-1 selection at cutoff 0.5.

## Local files

- `data/raw/imdb/dataset.zip`: original download (347,575,137 bytes).
- `data/raw/imdb/source-metadata.json`: original dataset's Kaggle metadata.
- `data/raw/imdb/kaggle-search-metadata.json`: supporting API response.
- `reports/data-audit.json`: validation results and archive SHA-256 checksum.

The archive contains two JSON Lines files (despite their `.json` extensions):

| File | Contents |
| --- | --- |
| `IMDB_reviews.json` | 573,913 reviews with `review_text`, Boolean `is_spoiler`, `movie_id`, `user_id`, `review_summary`, rating, and date |
| `IMDB_movie_details.json` | 1,572 movie metadata records, including plot summaries and synopses |

`is_spoiler=true` means spoiler; `false` means no spoiler according to the source label.
These are **whole-review labels**, not sentence labels. Do not copy a review's label onto every sentence or its title.

## Audit findings

- Spoilers: **150,924 (26.3%)**; no spoilers: **422,989 (73.7%)**.
- No empty review texts or missing required review fields.
- **564** repeated texts after case/whitespace normalization; **118** repeated rows disagree with the first occurrence's label.
- **2** review movie IDs do not have matching movie metadata.

Duplicates and inconsistent labels remain untouched. Before training, address these and
group splits by movie ID. Raw plot summaries are not labeled training examples.
Review labels alone will not establish performance on short web snippets or YouTube titles.

## Reproduce

Python 3.11+ and curl are sufficient; no ML dependencies are required.

```sh
python3 scripts/collect_data.py
python3 scripts/audit_data.py
```

Collection needs internet access. Auditing runs locally and streams records from the ZIP,
so extracting another roughly 967 MB is unnecessary. Collection reuses an existing ZIP;
the audit checks readability and records its checksum.

## Provenance and attribution

- Author: Rishabh Misra.
- Dataset: [IMDb Spoiler Dataset](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset), version 1.
- Citation: Misra, Rishabh. "IMDB Spoiler Dataset." 2019. DOI: 10.13140/RG.2.2.11584.15362.
- [Dataset paper](https://arxiv.org/abs/2212.06034).
- Kaggle's author-owned listing reports **Attribution 4.0 International (CC BY 4.0)**;
  this is recorded as source-reported licensing, not an independent rights determination.
- Raw data is excluded from version control. TV Tropes was downloaded for the
  audit above; Goodreads has not been collected.

Training dependencies are pinned in `requirements-training.txt`; data collection
and auditing require only standard-library Python and curl.

October 6: Gemini Flash title-only versus supplied-context development diagnostic stopped after repeated Google 503 service errors on the user-confirmed free-tier API project. Four of 207 rows saved in each arm; partial caches downloaded. Secrets remain inside Colab; no GPU inference or fine-tuning. Bounded retries were exhausted; completion monitor paused. Check `reports/gemini-context-comparison/PLAN.md` and `STATUS.json`; final metrics pending.

Gemini retry October 6: resumed saved answers with 30-second pacing, bounded 503 cooldowns and a five-attempt cap. Actual project quota is 20 requests/day; full comparison remains incomplete. See reports/gemini-context-comparison/PLAN.md. Higher-volume free API candidate: Groq GPT-OSS 120B, pending account/key and evaluation.

Gemini subsequently stopped at user request; completion monitor paused. Updated cooldown code saved but clean launcher not executed. Next requested work: Groq cost research.

Sentence/context RoBERTa preparation: seven TV-series source mappings cover 942 candidate sentences; a 200-train/60-dev preparation pilot is saved under ignored data/processed/sentence-context-pilot. The current draft summaries miss many episode events, so training remains unstarted. See reports/context-training-data-audit/PILOT.md for findings and episode-context follow-up.

Episode-context preparation is recorded in `reports/context-training-data-audit/EPISODE_CONTEXT.md`; current sentence data is TV Tropes, not Goodreads. The pilot remains pending grounding/label review; no new training started.

October 6: user authorized a bounded one-epoch hosted-GPU TV adaptation comparison using 744 train / 198 validation mapped candidates. See reports/tv-training/PLAN.md (relative to project root). Original preparation quality flags remain unchanged; this is exploratory corpus-label training, not reviewed gold.

October 6 experiment completed: one epoch each on Colab T4. Sentence-only catches 144/168 spoilers and flags 19/30 safe examples; context catches 168/168 but flags all 30 safe examples at cutoff .5. See reports/tv-training/RESULTS.md and comparison.json. No automatic second epoch.

### Expanded Qwen-alone evaluation (October 6, 2026)

Completed all 267 exploratory development examples on the user-confirmed Groq Free plan, with no training or purchases. Same Qwen3.8-27B prompt; 79 identical answers reused. Results: real content catches 5/5 spoilers, flags 1/138 safe, 19 abstentions; plot challenges catch 40/40, flag 2/20 safe; balanced TV markup catches 18/30, flags 8/30 safe, 10 abstentions. Labels are AI review or original markup, not human gold; do not treat aggregate scores as production accuracy. Updated quota pacing honors provider cooldowns; a specific JSON-generation failure is recorded as uncertainty instead of stopping unrelated rows. See [full results](reports/qwen-alone-expanded/RESULTS.md) and [plan](reports/qwen-alone-expanded/PLAN.md).

### Accurate-context TV diagnostic and scoring correction

Completed 14 targeted Qwen examples with manually checked episode context. Both arms classified all seven source-audited spoilers correctly; context resolved one vague safe reference. Three context passage quotes contained a control character in place of an apostrophe and failed evidence validation. No spoiler recall improvement demonstrated. Labels are a non-blind AI audit, not human gold. See [diagnostic results](reports/qwen-tv-verified-context/RESULTS.md).

Separately, whitespace-only rescoring of the original 60 TV examples restores eight invalid answers, including five spoilers: original-markup recall is 23/30 (76.7%) rather than 18/30 (60%). Original outputs and historical tables are preserved; this is a validator correction, not a model improvement.

### Qwen-only: 100 additional sentences

Completed 100 fresh corpus sentences across Firefly, Fringe, Sherlock, and Person of Interest, balanced 50 spoiler/50 safe by original markup. Qwen alone, with title and public premise but no retrieved passages, flagged 35/50 spoiler-labeled sentences and 14/50 safe-labeled sentences; 4 abstentions. Original corpus labels are noisy, not human gold; source partition was used for previous classifier training, so this is not a held-out cross-model evaluation. [Results and per-show breakdown](reports/qwen-alone-sentences-100/RESULTS.md).

### Qwen sentence error review

Reviewed all 31 mismatches/abstentions from the 100-sentence Qwen-only run. Found 13 dataset-policy disagreements, 5 model classification errors, 2 quote-validation failures, and 11 unresolved excerpts/policy boundaries. Non-blind AI audit only; no corrected overall-accuracy claim. [Full review](reports/qwen-alone-sentences-100/ERROR_REVIEW.md). Next: independent blinded review of all 100 with explicit policy boundaries, then locked scoring.

## Full prediction-hidden AI review

Two reviewers independently labeled disjoint sets of 50 anonymized inputs without Qwen predictions or original source labels. Fixed policy; all labels locked before scoring. References: 38 spoiler, 45 safe, 17 unresolved. Qwen caught 30/38 spoilers (78.9% recall), falsely flagged 8/45 safe sentences (17.8%), and matched 67/83 resolved references (80.7%), including abstentions as non-correct. Precision 78.9%. One abstention on resolved references. The 17 unresolved references are excluded from binary scores. No plot-source verification was used in this review; these are AI policy judgments, not human gold. Original scores remain preserved. [Full review](reports/qwen-alone-sentences-100/blind-review/RESULTS.md).

## Paused hybrid RAG comparison

Paused at explicit user request with 8/100 RAG answers saved; baseline and locked labels intact. No automatic continuation. Cumulative caps: 120 request attempts and 100000 tokens; resume only on user request. [Resume instructions](reports/qwen-locked-hybrid-rag/RESUME.md).
