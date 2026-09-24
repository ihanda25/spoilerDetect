# Full-data MiniLM, take 2 (no windowing) — plan and status

This file is the durable source of truth for this experiment. It exists so that if
this conversation/session is lost, a fresh session (or the user) can read this file
alone and know exactly what to do next. Update the **Status** section every time a
step below is completed.

## Why this replaces the original `reports/full/` attempt

1. The original full-data plan (`reports/full/FULL_FINETUNING_RESULTS.md`) covered
   every token of every review via overlapping 512-token windows (64-token overlap)
   with review-level max-pooling. That was started, and its own measured in-run ETA
   (`reports/full/status.json`) was **~17 days per epoch** on this Mac's MPS backend.
   It was deliberately killed (`kill -TERM`) before any checkpoint was written —
   nothing was lost, but the approach was abandoned as impractical locally.
2. ModernBERT-base (149M params, native 8192-token context, would have removed
   windowing entirely) was tried next as a replacement. A real calibration run
   **OOM'd** on this machine (`MPS backend out of memory`, 24GB unified memory total)
   even at batch size 8, because ModernBERT's global-attention layers are quadratic
   in sequence length and this Mac doesn't have enough headroom for that plus
   gradient-checkpoint-free full-length batches. It was fixed (gradient checkpointing
   + bf16 + length bucketing) but was never pursued further once the simpler option
   below was measured and found sufficient — ModernBERT-base weights have been
   deleted from `models/base` to save space; nothing else references it.
3. The actual fix: **drop windowing, keep MiniLM, cap at a plain 512-token
   truncation** (no overlapping windows, no max-pooling, one forward pass per
   review). Per `reports/full-data.json`, only 17.4% of train reviews exceed 512
   tokens at all (median is 240 tokens), so truncation at 512 loses the tail of a
   minority of reviews, not most of them. A real measured calibration (MiniLM-L12,
   batch 16, fixed 512-token padding, real review text, real MPS forward+backward+
   optimizer step) gave a **stable, reproducible ~0.596s/batch (16 reviews)**, i.e.
   **~0.0373 sec/review**. Extrapolated to the full 444,644-review train split:
   **~4.6 hours/epoch (train only)**, ~10 hours all-in for 2 epochs, ~14–15 hours
   all-in for 3 epochs, including per-epoch validation passes and the final test
   pass. This is the plan being executed.

Do not re-litigate this decision without new measured evidence — re-run a small
real calibration (see pattern in git-free history: this file) rather than guessing.

## Method

- Model: `microsoft/MiniLM-L12-H384-uncased`, pinned revision
  `44acabbec0ef496f6dbc93adadea57f376b7c0ec` (same checkpoint as the pilot).
- Data: reuse the already-tokenized, untruncated, movie-disjoint full split at
  `data/full/{train,validation,test}.{tokens.bin,offsets.npy,labels.npy,...}`
  (produced by `scripts/prepare_full.py`; do not re-run it, it already exists and
  is correct — 444,644 / 63,836 / 64,405 reviews, same split as the pilot).
- Sequence handling: truncate every review to the first 510 text tokens (+
  `[CLS]`/`[SEP]` = 512), **no overlapping windows, no max-pooling** — one review,
  one forward pass, one label.
- Batch size 16, fixed `padding='max_length'` to 512 (matches the exact
  calibration that was measured — do not switch to dynamic padding without
  re-measuring; fixed padding was chosen deliberately for predictability after
  several surprises in this project, not because it's fastest).
- AdamW, lr 2e-5, weight decay 0.01, linear warmup (10%) + linear decay, gradient
  clipping at 1.0 — same hyperparameters as the pilot (`scripts/finetune.py`).
- 3 epochs planned, checkpoint selection by validation F1 (same as pilot).

## Execution mode — IMPORTANT, this is a user requirement

**Run exactly one epoch per invocation, then stop**, UNLESS the auto-continue
policy below says to proceed. Do not loop over multiple epochs automatically
beyond what that policy allows, and do not exceed 3 total epochs under any
circumstance without new, explicit user instruction. Each run:

1. Loads the checkpoint if one exists (resumes from the next epoch), else starts
   fresh from the pretrained base checkpoint.
2. Trains exactly one full epoch over the 444,644-review train split.
3. Runs validation (63,836 reviews), logs metrics, saves a checkpoint to
   `models/minilm-full/checkpoint.pt` (model + optimizer + scheduler + epoch
   number + best-so-far validation F1 + rng state), and keeps the best-F1 model's
   weights separately in `models/minilm-full/best/`.
4. Updates `reports/full-v2/status.json` and appends to
   `reports/full-v2/training.log`.
5. **Exits the process.** Waits for the user's go-ahead before the next epoch.

After the final planned epoch (3), also run the test-set before/after comparison
and write `reports/full-v2/FULL_V2_RESULTS.md` (mirroring
`reports/FINETUNING_RESULTS.md`'s format).

### Auto-continue policy (user-authorized 2026-09-24, NARROWED 2026-09-24)

**Superseded for epoch 2→3: the user asked to decide epoch 3 personally.**
Epoch 1→2 auto-continue already happened (see Status below) under the original
rule. That rule no longer applies past epoch 2. Current rule:

- After epoch 2 completes: **always stop, do not start epoch 3 automatically**,
  regardless of the F1 threshold or trend. Send the user a push notification
  that epoch 2 is done (they explicitly asked to be pinged) and wait for them
  to explicitly say whether to run epoch 3.
- Only start epoch 3 when the user explicitly asks for it in that later
  conversation/session.
- If somehow resuming after epoch 1 only (e.g. a lost session before epoch 2
  started): the original epoch 1→2 auto-continue rule (F1 < 0.65, or epoch 1)
  still applies to get to epoch 2. It is specifically epoch 2→3 that now
  requires explicit user sign-off.

Original rationale for the 0.65 F1 threshold (still relevant context for the
user's own decision on epoch 3, just no longer an automatic trigger): the pilot
(`reports/FINETUNING_RESULTS.md`) reached F1=0.509 with 55x less training data
and a harsher 192-token cutoff. Also worth the user's attention when deciding:
accuracy is a misleading metric on this dataset (73.7% non-spoiler baseline
already scores ~74% accuracy predicting nothing); recall and F1 are what
actually matter, and even a fully-trained text-only classifier likely has a
real ceiling somewhere around F1 0.5-0.7 / ROC-AUC 0.85-0.92 based on published
results on this exact dataset — if the gap after epoch 2/3 isn't closing, the
next lever discussed is plot-summary RAG (see conversation / consider adding
that discussion to this file if it becomes the active plan), not more epochs.

Rationale for the 0.65 F1 threshold: the pilot (`reports/FINETUNING_RESULTS.md`)
reached F1=0.509 with 55x less training data and a harsher 192-token cutoff;
0.65 is a meaningfully higher bar appropriate for the full dataset, without
being high enough to force all 3 epochs regardless of what validation shows.
This value can be revisited if it turns out to be miscalibrated once epoch 1's
real numbers are in — if you're a future session reading this and the number
looks wrong in hindsight (e.g. F1 plateaus around 0.55 for legitimate reasons),
use judgment and say so to the user rather than blindly following a stale
threshold.

Command to run one epoch:
```sh
.venv/bin/python scripts/full_finetune_v2.py
```
It is idempotent/resumable: re-running it after a completed epoch just does the
next one. If interrupted mid-epoch, re-running restarts that epoch from scratch
(no mid-epoch checkpointing — an epoch is ~4.6h, short enough that this is fine).

## Status

- [x] Epoch 1/3 — complete 2026-09-24 05:29 -0400. Validation F1=0.5602, accuracy=0.792,
      precision=0.659, recall=0.487, roc_auc=0.808, average_precision=0.643. Took
      ~5.4h wall-clock this run (vs. ~4.6h measured/expected) because the Mac went to
      sleep partway through and paused the process for a stretch before `caffeinate
      -w <pid>` was attached to fix it — see note below. Below the 0.65 auto-continue
      threshold and this was epoch 1, so epoch 2 was started automatically per the
      auto-continue policy above, without waiting for the user.
- [x] Epoch 2/3 — complete 2026-09-24 10:19 -0400. Validation F1=0.5336 (DOWN from
      epoch 1's 0.5602), precision=0.727 (up from 0.659), recall=0.422 (down from
      0.487), accuracy=0.800, roc_auc=0.815, average_precision=0.660. F1 regressed
      because recall dropped more than precision gained — the ranking metrics
      (roc_auc, AP) still improved, so scores may just need a lower decision
      threshold rather than more training, but at the default 0.5 cutoff this is a
      plateau/regression, not an improvement. `models/minilm-full/best/` was NOT
      overwritten — it still holds epoch 1's checkpoint (best_f1=0.5602). Per the
      user's explicit request, epoch 3 was NOT auto-started; the user will decide
      whether to run it after reviewing these numbers.
- [ ] Epoch 3/3 — awaiting explicit user decision (not automatic — see policy above).
- [ ] Final test evaluation + `FULL_V2_RESULTS.md`

### Operational note: keep the Mac awake

Epoch 1 stalled for a long stretch because nothing was preventing system sleep
when it was launched. Fix applied: `caffeinate -dimsu -w <training_pid>` was
started (detached from the shell, so it survives independently) right after
epoch 1 stalled, and should be (re-)started the same way immediately after
launching each subsequent epoch, tied to that epoch's process PID. Caffeinate
cannot override a physically closed laptop lid without an external display —
if that's blocking progress, only the user can fix it (open the lid / plug in
a display), it's not something a script can work around.

Last updated: 2026-09-24 05:29 -0400. `models/base` holds the MiniLM-L12
checkpoint. `scripts/full_finetune_v2.py` and
`scripts/full_finetune_v2_final_eval.py` both exist and the former has now run
successfully once. Trust `reports/full-v2/status.json` and this Status section
over any other point-in-time claim in this file.
