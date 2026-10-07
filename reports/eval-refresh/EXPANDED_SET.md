# Expanded AI development evaluation set

Completed 2026-10-05. User authorized assistant/agent judgments. No model inference, threshold tuning, training or GPU provisioning performed in this step.

## Current files

- `reviewed-dev-v2.jsonl`: 207 development records, explicitly AI-reviewed; annotations, rationales, revealing quotes, relevance and source evidence retained.
- `blind-inputs-v2.jsonl`: exactly id/text/surface/split, accepted by the existing text-only diagnostic loader. No labels, rationales, plot evidence or reviewer data in model inputs.
- `input-metadata-v2.json`: work/group/slice lookup for future context retrieval and separate slice reports. Do not send this to the text-only baseline.
- `summary-v2.json`: counts, fixed RoBERTa settings, source quote budgets and limitations.
- `source-challenge.jsonl`: the new, source-derived challenge slice only.
- `source-verification.json`: parent checks for every challenge excerpt, source URLs, excerpt hashes and verification times.
- `context-adjudications.json` and `CONTEXT_REVIEW.md`: parent changes with prior judgments preserved in the combined export.

| Slice | Total | Spoiler | Safe | Uncertain |
|---|---:|---:|---:|---:|
| Original real-content dev, refreshed AI judgments | 147 | 5 | 138 | 4 |
| Additional sourced plot challenge | 60 | 40 | 20 | 0 |
| **Total** | **207** | **45** | **158** | **4** |

The challenge covers 20 additional works: 15 films and five books. Two positives per work are correlated, not 40 independent stories. It was intentionally selected to expose later events, identities, fates and outcomes; the negative examples are brief setup/character descriptions. These are short plot-summary fragments, not newly collected user reviews/comments or real YouTube captions. They do not replace the original product-surface slice or establish production prevalence. No new television-series or ASR coverage was added.

## Annotation and source checks

The first 35 examples were reviewed by the parent assistant; gpt-6-luna reviewed the remaining 112 individually with distinct rationales. Parent review corrected an unidentified-work claim, verified several premise boundaries, and retained uncertainty where causal/outcome details were not verified. Original human/AI decisions remain in their previous files. Refreshed labels are AI judgments, not human corrections or gold.

The challenge agent's draft included inaccurate paraphrases presented as quotations and some ambiguous selections. Those were not accepted as source-verified data. The draft was repaired, then the parent independently read all 20 source pages and checked every retained fragment for a contiguous match after whitespace and typographic normalization. Further parent edits replaced remaining nonmatches, removed an opening-scene death used as a later-event spoiler, and replaced the famous Jekyll/Hyde identity with a later character fate. Each final example has an individual rationale. Source URLs and text hashes bind the checks to the retained text; matching a Wikipedia synopsis is not independent proof of every canonical plot fact.

At most 25 distinct excerpt words are retained per source page. The pages are Wikipedia summaries, not IMDb/TVTropes or a newly licensed open corpus of reviews. The excerpts preserve the source wording with minor typographic normalization recorded in the verification file. No full copyrighted pages were saved. The candidate text is intentionally short, often a clause; references to pronouns may need supplied work context. Inspect performance by slice and quote length rather than treating this as a full-review benchmark.

The original 98 held-out candidate rows are excluded and untouched. Additions were checked against declared held-out work keys and source groups using grouping metadata only. Original grouping remains semantically unverified, so this is not proof of exhaustive franchise/alias independence or freedom from pretraining contamination. Popular works may be familiar to an LLM already.

## Rebuild and next comparison

Run `python3 scripts/build_eval_refresh.py` to validate/rebuild identical exports. Changed existing outputs are refused; version later edits. `python3 -m unittest discover -s scripts -p test_eval_refresh.py` checks dev/text/group preservation, AI provenance, quote evidence, uncertainty gaps and duplicate rejection.

Fixed baseline: `Zritze/imdb-spoiler-robertaOrigDatasetLR1`, revision `56fee120f8495ccfc5001e3fbd1478656d17001a`, spoiler index 1, threshold 0.5, first 512 tokens, text only. Existing scores use older labels and cannot be presented as results on v2. V2 results are now saved and verified in `reports/context-comparison/RESULTS.md`.

Next compare fixed RoBERTa, the same LLM without context, and the same LLM with sourced work-level plot context. Report original-content and challenge metrics separately, with minor events/major twists distinguished. Count abstained positives as missed spoilers in overall recall and report coverage, false alarms and latency. Do not use reviewer rationales or selected positive snippets as label-conditioned context. Prepare and freeze work-level context independently, then use the same context for every example from that work.

The existing three-model text-only runner can load the blind inputs; its report command still needs an annotation mapping and saved model predictions. Use an explicitly AI-diagnostic report, with per-slice summaries added before claiming a comparison. Context inference support and 23 frozen work summaries are now implemented in `scripts/compare_context_models.py` and `reports/context-comparison/`; a three-arm GPU diagnostic completed on Colab and results were saved locally. See that directory for current status and context coverage. Strict human-gold/test evaluation gates remain unchanged; do not bypass them to score this AI pack. No heavy fine-tuning is needed for the next inference experiment, and no new paid resources/API costs are assumed.
