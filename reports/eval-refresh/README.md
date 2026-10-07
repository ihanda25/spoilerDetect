# Evaluation refresh, then context-assisted LLM

User approved this sequence on 2026-10-05. No training or GPU run started.

**Latest completed state:** evaluation v2 now contains 207 explicitly AI-reviewed development examples: 45 spoiler, 158 safe, 4 uncertain. All original 147 dev rows were re-reviewed, with sourced adjudication where possible; a separate 60-row challenge adds 40 spoilers across 20 works (15 films, 5 books). All retained challenge quotes were checked against source pages. Read `EXPANDED_SET.md` and `summary-v2.json` for current files, verification, limitations and next inference work. Earlier sections below preserve the staged plan; references to remaining 112 rows being pending are historical. Held-out test unchanged; no new model run or training.

## Latest authorization and first-batch results

The user explicitly delegated judgments to the assistant or an agent: “Just make the decisions on your own or with an agent. I trust you.” AI review is now authorized for the exploratory development comparison; do not wait for human labels to perform that diagnostic. This does not change annotation provenance or the strict human-gold/test scorer. Report AI agreement separately and preserve original human decisions.

`first-batch-ai-decisions.jsonl` contains 35 individual AI judgments under the refreshed policy: 4 spoiler, 26 safe, 5 uncertain. Every decision includes a distinct reason; spoilers have exact revealing quotes; uncertainties identify missing context. No new plot-source verification was performed. `scripts/annotate_eval_first_batch.py` records these individually written judgments and refuses to overwrite an existing export. All grouping/eligibility flags remain pending, and the other 112 development examples have not been re-reviewed under this policy.

Of the six previously disputed human positives, five were assessed safe (opinion, general descriptions, or viewing availability); the breakup description remains uncertain because its premise boundary is not verified. These are revised AI judgments, not corrections to human gold. Four new-policy positives describe Inception's dream-level/kick sequence and Breaking Bad's payment offer/Fring encounter. They are concentrated in two works, so this batch still cannot establish broad recall. It was selected using old annotation categories and must not be scored as a representative random sample.

Next: resolve the five context gaps with sourced public premises/plot passages, review the remaining dev examples, and collect additional development positives across more works/surfaces. Keep source evidence separate from labels. Then run the fixed RoBERTa and LLM context arms as explicitly AI-labeled diagnostics, reporting exclusions and coverage; do not enable locked test or bypass the human-gold gate.

## Label policy

Assume the reader has not started the work and knows its public premise. A spoiler reveals a later plot event, outcome, twist, hidden identity, or character fate beyond that premise. Opinions, cast/production news, vague references to an ending, and public premises alone are safe. An opinion can still contain a spoiler if it reveals an event. Do not assume that every relationship breakdown or death mentioned is a later revelation: establish the premise and the work first.

Judge only the displayed text. If the work, event, premise boundary, or surrounding context is unknown, use uncertain and state what information is missing. Quote the exact revealing span and explain which event it reveals for every positive. For safe decisions explain why no later event is revealed. Resolve relevance separately: irrelevant industry news is not useful evidence of story-spoiler detection, even when safe. Preserve original decisions; new policy decisions are a separate version.

## Ready for human review

- `first-batch.jsonl`: 35 blinded development examples, selected from prior human decisions and AI positive/uncertain decisions. Review queue only; selection is biased and cannot be used as a standalone accuracy benchmark.
- `dev-review.jsonl`: all 147 development excerpts, with fresh blank decisions and no prior labels or model predictions. The first batch is a subset; merge its exported decisions by ID before reviewing the remaining rows.
- `manifest.json`: original pack hash, unchanged dev IDs/text hashes/groups, and review selection provenance. The 98 held-out test examples are not included or changed.

Open `../../tools/review_spoiler_examples.html`, load the first batch, enter a reviewer ID, and download decisions regularly. Save the export alongside this directory with a new filename. Require a reason for every decision. Check the grouping box only after actually checking work aliases and secondary references. The original human/AI annotations remain in `../real-review-pack/`; do not replace them. AI assistance remains explicitly AI until a human independently reviews it.

These dev-only files are annotation inputs, not inputs to the strict scorer, which requires the complete dev/test dataset. Before scoring, merge exports into a versioned complete pack by ID, verify unchanged text hashes, preserve pending held-out rows, and apply existing review/group gates. Do not bypass these gates or mark unresolved context as safe.

## Expand coverage before interpreting recall

Target 100–200 human-reviewed, relevant examples with at least 40–50 definite spoilers and a comparable set of difficult safe examples. This is a diagnostic target, not production prevalence. The current 147 examples do not meet it. Collect additional real development text independently of model scores; freeze source/work grouping before labeling. Include subtle paraphrases, premise-only descriptions, opinions mentioning endings, and spoilers late in longer text. Cover books, films/shows, comments, reviews, headlines, and actual YouTube captions where available; preserve timestamps, provenance and excerpt transformations. The existing film subtitles are not broad YouTube transcript coverage.

Keep a challenge slice separate from naturally sampled text so balanced collection does not imply real-world precision. Document exclusions, uncertain labels and per-surface denominators. Keep newly collected work/source groups disjoint from held-out groups. The six disputed human positives must be reviewed without assuming model consensus is correct. Use a second human reviewer for disagreements when possible. Do not fabricate human judgments or force positive counts.

## Fixed RoBERTa baseline

Model `Zritze/imdb-spoiler-robertaOrigDatasetLR1`, revision `56fee120f8495ccfc5001e3fbd1478656d17001a`; spoiler class index 1, threshold 0.5; text only, first 512 tokens as in the completed comparison. Retain that configuration for the first comparison. Any windowed variant or tuned threshold is a separately named development experiment. Report truncation counts; long passages can expose this baseline's limits.

Existing `../three-model-comparison/results.json` is exploratory agreement with the old labels, not gold under the refreshed policy. Recompute after review rather than reusing old scores as current evidence.

## Next experiment: LLM with supplied plot context

After labels and relevant work identities are ready, compare the fixed RoBERTa baseline, the same LLM without context, and the same LLM with sourced plot passages on identical examples. First supply context directly to separate context quality from retrieval failures. Freeze prompt, context budget and passages on dev; never construct context from labels, reviewer rationales or model errors. Record source URL, work/edition, passage text/hash and missing-context cases. Use the same label policy for every arm.

Request safe/spoiler/uncertain plus the candidate's revealing span and a matching plot passage. Treat abstentions as missed positives in overall recall, and separately report coverage, false alarms, precision, per-surface metrics, latency and context availability. Do not hide failures behind answered-only accuracy. No test-driven prompt changes. Existing strict final-test gating remains in force; a final test is deferred until choices are frozen and human/group review is complete.

If supplied context helps, build retrieval and evaluate retrieval coverage separately. No additional fine-tuning is scheduled. Notify the user before heavy training; GPU spending/resources must follow the user's current authorization, with no new purchases or paid APIs assumed for this future experiment.
