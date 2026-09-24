# Full-data MiniLM, v3 — review + plot_summary pairs

This file is the durable source of truth for this experiment, same convention as
`../full-v2/PLAN.md` and `../full/RESUME.md`: written so a fresh session with no
memory of prior conversations can read this file alone and know what to do next.
Update the **Status** section every time a step below is completed.

## Where this fits in the project's history

1. **Pilot** (`../FINETUNING_RESULTS.md`): MiniLM-L12, 8,000 train reviews, 192-token
   truncation. F1=0.509.
2. **v1, abandoned** (`../full/`): same model, all 444,644 train reviews, full
   coverage via overlapping 512-token windows + max-pooling. Killed before any
   checkpoint saved — its own measured ETA was ~17 days/epoch on this Mac's MPS.
3. **ModernBERT, abandoned**: tried as a way to avoid windowing (native 8192-token
   context). OOM'd on this machine's 24GB unified memory during calibration.
   Weights deleted; not pursued further once v2 below was measured and found
   sufficient.
4. **v2, done through epoch 2, paused for a decision** (`../full-v2/`): same
   MiniLM, all 444,644 train reviews, but a plain 512-token truncation instead of
   windowing (no overlapping windows, no max-pooling — one review, one forward
   pass). Real measured cost ~4.6-5.4h/epoch. Results:
   - Epoch 1: F1=0.5602, precision=0.659, recall=0.487, roc_auc=0.808, AP=0.643.
     Best checkpoint (`../../models/minilm-full/best/`).
   - Epoch 2: F1=0.5336 (regressed), precision=0.727 (up), recall=0.422 (down),
     roc_auc=0.815 (up), AP=0.660 (up). Ranking metrics kept improving but the
     fixed 0.5 threshold landed worse. Did not overwrite the epoch-1 checkpoint.
   - Epoch 3 was available but the user chose NOT to run it, opting instead to
     try this v3 (plot-summary) approach, on the reasoning that review-only text
     appears to be nearing its ceiling (recall stuck high-0.4s across 2 epochs)
     and plot-summary context is more likely to move the number that matters
     (recall) than a 3rd epoch of the same setup would.
5. **v3, this experiment**: same MiniLM checkpoint, same 444,644 reviews, same
   "plain truncation, no windowing" philosophy, but each input is now a
   BERT-style pair — `[CLS] plot_summary [SEP] review [SEP]` — instead of review
   text alone. Rationale from the conversation that led here: a text-only
   classifier structurally cannot tell "this sentence describes a real plot
   event" from "this sentence is vague foreshadowing" — it has no access to what
   actually happens in the movie. Giving it the movie's own `plot_summary` (from
   `IMDB_movie_details.json`, always present for 1,570/1,572 movies, present for
   all but 2 review `movie_id`s) as context directly targets that gap. This is a
   direct lookup by `movie_id`, not search/retrieval over a corpus — "RAG" is
   not quite the right term for it, even though the idea originated from a RAG
   discussion.

   Why this doesn't just memorize training movies: evaluation uses the same
   movie-disjoint splits as every prior experiment (`split_for()` in
   `prepare_training.py`/`prepare_full.py`, bucketed by SHA256 of `movie_id` —
   none of the ~164 validation or ~164 test movies appear in the ~1,244 training
   movies). If the model were just memorizing training-movie plot/review
   associations rather than learning the general "does this text describe an
   event in this summary" skill, that would show up immediately as a validation/
   test recall collapse — the same mechanism that already caught v2's epoch-2
   regression would catch this too.

## Method

- Model: `microsoft/MiniLM-L12-H384-uncased`, same pinned revision as before
  (`models/base`).
- Data: reuses `data/full/{split}.tokens.bin` (review text, untruncated, already
  tokenized) AND `data/full/{split}.movie_ids.npy` (already saved by
  `prepare_full.py` — no new data-prep step or re-tokenization needed for review
  text). Plot summaries are loaded fresh each run directly from
  `data/raw/imdb/dataset.zip`'s `IMDB_movie_details.json` and tokenized once at
  startup (1,572 movies, a few seconds) — see `load_plot_summaries()` in
  `scripts/full_finetune_v2_plot.py`.
- Input format: `[CLS] plot_summary(<=224 tok) [SEP] review(<=285 tok) [SEP]`,
  fixed-padded to 512 total, with proper `token_type_ids` (0 for the
  `[CLS]+plot+[SEP]` segment, 1 for the `review+[SEP]` segment) so the model can
  structurally distinguish the two spans (standard BERT pair-classification
  usage — MiniLM-L12 has `type_vocab_size=2`, i.e. it supports this).
  - `PLOT_MAX=224`: measured with the real tokenizer over all 1,572 movies —
    mean 132 tokens, median 123, p90 216, max 267. 224 covers the p90 fully and
    only truncates the handful of movies above it.
  - `REVIEW_MAX=285` (= 512 − 3 special tokens − 224): smaller than v2's 509,
    so reviews lose more tail content than in v2. This is the direct tradeoff
    of adding plot context inside the same 512-token budget.
  - Encoding logic was sanity-checked on real data before launching training
    (correct segment boundary, correct token_type_ids, plot text decodes
    correctly for a spot-checked movie) — see conversation for the verification
    output; not re-derived here.
- Same hyperparameters as v2: batch 16, fixed `padding` to 512 always (not
  dynamic — same predictability reasoning as v2), AdamW lr=2e-5, weight decay
  0.01, linear warmup (10%) + linear decay over 3 planned epochs, gradient
  clipping 1.0, checkpoint/best-model selection by validation F1.
- Expected cost: about the same as v2 per epoch (~4.6-5h) — same model size,
  same batch size, same total 512-token budget per example, just a different
  split of what's inside those 512 tokens. Adding plot summaries does not
  meaningfully change memory or compute cost.

## Execution mode — same user requirement as v2, even stricter here

**Run exactly one epoch per invocation, then stop. Do NOT auto-continue to the
next epoch under any circumstances for this v3 experiment** — the user
explicitly narrowed v2's auto-continue policy to require their sign-off past
epoch 1, and for v3 they asked to see epoch 1's result before deciding anything
further. Treat every epoch beyond the first as requiring an explicit, fresh
"go ahead" from the user, not a threshold-based automatic decision. Ping the
user (push notification) when an epoch completes, the same way as v2.

Command to run one epoch:
```sh
.venv/bin/python scripts/full_finetune_v2_plot.py
```
Resumable/idempotent the same way as v2 (checkpoint at
`models/minilm-full-plot/checkpoint.pt`). Attach `caffeinate -dimsu -w <pid>`
(detached, e.g. `nohup caffeinate -dimsu -w <pid> >/dev/null 2>&1 & disown`)
immediately after launching — v2's epoch 1 stalled for hours because this was
forgotten the first time; don't repeat that.

Final eval (only once all planned epochs are done and the user has decided to
stop): `.venv/bin/python scripts/full_finetune_v2_plot_final_eval.py`. It
refuses to run unless `status.json` shows the last epoch as complete, and
automatically pulls in v2's test-set numbers for a side-by-side comparison if
`reports/full-v2/results.json` exists.

## Status

- [x] Epoch 1/3 — complete 2026-09-24 16:02 -0400. Validation F1=0.5558,
      accuracy=0.765, precision=0.573, recall=**0.540** (up from v2's 0.487 —
      the specific weakness this experiment targeted), roc_auc=0.788 (down from
      v2's 0.808), average_precision=0.599 (down from v2's 0.643). Mixed result:
      recall improved as hoped, but precision/AUC/AP regressed and F1 is roughly
      flat vs. v2 epoch 1 (0.556 vs 0.560). Plausible causes: the model may be
      more liberal about flagging plot-overlap as a spoiler even when it isn't
      one, and/or the review portion is more truncated here (285 tok vs. 509 in
      v2) so some discriminating review-language signal is lost. Training was
      paused (SIGSTOP) once near the very end (step 26800/27791) when the user's
      laptop battery was critically low, then resumed (SIGCONT) once they found
      a charger — no progress lost, see git-free history: this was step-for-step
      the same recovery mechanism as v1's original pause design.
- [ ] Epoch 2/3 — awaiting explicit user decision (not automatic — see policy above).
- [ ] Epoch 3/3 — requires explicit user go-ahead after seeing epoch 2.
- [ ] Final test evaluation + `FULL_V2_PLOT_RESULTS.md`.

Last updated: 2026-09-24 16:02 -0400. Trust `reports/full-v2-plot/status.json`
over this paragraph if they disagree.
