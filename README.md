# Local spoiler detection with MiniLM

## Project history and current state (read this first)

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
   opting instead to try v3 below since review-only text looked like it was
   nearing its ceiling (recall stuck in the high-0.4s).
4. **v3, active** — `reports/full-v2-plot/`. **Read
   `reports/full-v2-plot/PLAN.md` first.** Same MiniLM, same 444,644 reviews,
   same "plain truncation, no windowing" approach, but each input is now a
   pair — `[CLS] plot_summary [SEP] review [SEP]` — instead of review text
   alone, so the model can compare the review against what's actually known to
   happen in that movie (direct lookup by `movie_id`, not corpus retrieval).
   Same movie-disjoint splits as every prior experiment, so this is a fair,
   non-memorizing test of whether plot context helps.

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
run offline. The saved model is in `models/minilm-spoiler`.

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
- Raw data is excluded from version control. Goodreads and other datasets have not been collected.

Training dependencies are pinned in `requirements-training.txt`; data collection
and auditing require only standard-library Python and curl.
