# Resume real-content evaluation

Updated 2026-10-05. No training started. No real-data threshold or test result exists.

## Completed

- Colab T4 synthetic diagnostics: RoBERTa truncation caught 15/20 provisional spoilers, windows 19/20, both 8 false positives. See ../colab-roberta/RESULTS.md. Not human-reviewed accuracy.
- 245 real-text candidates, 147 dev / 98 test. See ../real-review-pack/README.md for provenance, sample bias and limitations. The immutable candidates remain unlabeled. Ishaan has supplied 31 browser decisions, backed up separately; a cheaper gpt-6-luna agent labeled the remaining 214 as AI annotations. Grouping still requires semantic verification. Review entries are brief standfirst excerpts, not long reviews; transcript entries are subtitles from two films, not YouTube ASR output.
- scripts/product_eval.py: all-token overlapping windows, max aggregation, per-record atomic cache, fingerprint checks, human-review gates, dev-only joint configuration/threshold selection, frozen test selection and refusal to overwrite changed test results.
- notebooks/Spoiler_Real_Content_Evaluation.ipynb embeds the evaluator source and expects a complete reviewed JSONL upload. It compares RoBERTa truncate/window on dev. Test is an explicit final cell, disabled by default. No Drive mount, fine-tuning or paid resource provision.

## Required next human work

Open tools/review_spoiler_examples.html directly in a browser (or use the local server). Load reports/real-review-pack/candidates.jsonl. Enter reviewer ID, choose safe/spoiler/uncertain, and download decisions regularly. No judgments are saved to disk until exported. The agent must not fill out human-reviewed fields itself.

Resolve unknown work names, aliases and secondary references before marking a group verified. If group assignments need changing, preserve the original pack and create a revised manifest before scoring. The checkbox is an attestation, not an automated semantic leakage check. Null uncertain labels are excluded from binary metrics. The workflow refuses pending review and grouping flags.

The full reviewed file must retain dev and test records. Do not upload dev.jsonl alone. Do not treat omitted or irrelevant examples as safe; document exclusions/uncertainty. Never infer real-content accuracy from the synthetic run.

## GPU run

Upload notebooks/Spoiler_Real_Content_Evaluation.ipynb through Colab File > Upload notebook; connect a T4. Upload the complete reviewed decisions when prompted. Optional saved cache ZIP resumes completed records only when exact data/model/runtime/config fingerprints match. Runtime files are ephemeral; download the archive after each stage. The notebook has been syntax checked; real-data GPU execution awaits completed human review.

Use development metrics to choose truncation/windowing and its threshold. Inspect and freeze selection.json, then enable RUN_LOCKED_TEST for the single selected configuration. Do not tune again after test. Report per-surface denominators and exclusions. A higher score on this short-excerpt convenience sample cannot establish full-review/video quality.

## MiniLM comparison

The same CLI accepts the owned epoch-2 checkpoint via --model models/base --checkpoint models/minilm-full/checkpoint.pt. Use the same reviewed file and separate dev caches/selections. Do not accidentally use models/minilm-full/best (epoch1). For a hosted run, package the owned model/tokenizer/checkpoint and transfer it separately; the current RoBERTa notebook intentionally contains no 382 MB checkpoint. A real-content MiniLM comparison has not run yet. No need to retrain either model.

## Rebuild and checks

python3 scripts/build_product_colab.py regenerates the notebook from the evaluator.
python3 -m unittest discover -s scripts -p test_product_eval.py checks threshold boundaries, leakage, evidence gates, cache integrity and frozen test behavior. No model training is involved.

## Annotation handoff (2026-10-05)

User requested a cheaper model agent to finish the remaining labels. Preserve the first 31 human decisions in `../real-review-pack/human-decisions-backup.jsonl`; the browser showed 31 saved decisions, although the request originally said 20. Labels were captured from visible selected controls without changing decisions. Original review timestamps were not exposed by the page, so `reviewed_at` remains null and `decision_captured_at` records backup time. Do not fabricate original timestamps; these backups intentionally cannot pass the strict scorer evidence gate. All 31 group checkboxes were unchecked. The original live tab remains available for exporting its complete metadata.

AI sidecar: `../real-review-pack/ai-annotations.jsonl`, model gpt-6-luna, rows 32–245. Keep `label_provenance=ai` and `review_status=ai_reviewed`; never convert these to human/corpus labels to satisfy evaluation gates. Any metrics using them would be preliminary agreement with AI judgments, not human-validated accuracy. The current evaluator/notebook does not accept AI labels. No training or real-content scoring started.

Completed combined file: `../real-review-pack/annotated-human-and-ai.jsonl` (31 human, 214 AI). Final AI labels: 183 safe, 2 spoiler, 29 uncertain. Distinct per-example rationales; parent spot-check correction audited in metadata. This convenience pack has very few AI-labeled spoilers; it cannot support a strong recall estimate without further human verification/positive examples.

## Active next step: three-model LLM-alone baseline

User authorized MiniLM/RoBERTa/LLM-alone comparison and chose a small open LLM on Colab. `../three-model-comparison/README.md` is the run handoff. The runner uses only development excerpts and fixed existing classifier cutoffs; strict human test gates are unchanged. Gemini key was offered but is unnecessary for this run.

User cost constraint (2026-10-05): zero spending. Free Colab resources only; no paid API requests, subscriptions, compute-unit purchases, or paid runtime fallback. Stop if free resources are unavailable.

Three-model run stopped: existing Colab Pro usage was discovered; current runtime deleted with explicit user approval. No new purchases or API charges; existing compute-unit consumption cannot be ruled out. See `../three-model-comparison/partial-results.json` and README. LLM baseline remains incomplete. A second active session was reported but not modified. Do not restart a hosted GPU without verifying zero paid-unit consumption.

LLM baseline ON HOLD by explicit user request (2026-10-05). Resume only when the user asks; do not automatically launch or reconnect. Existing MiniLM/RoBERTa results remain saved. Zero spending remains required.

Latest steering: user requested finish LLM test and explicitly authorized existing Colab units, no purchases. LLM hold revoked for this comparison. No paid API calls or credit/subscription purchases. Resume Qwen only and restore classifier caches.

Three-model development diagnostic COMPLETE. Current results: `../three-model-comparison/results.json`; Qwen106/147 uncertain, RoBERTa2/2 AI positives with2false alarms, MiniLM0/2. Human-policy disagreement requires review. No test evaluation, training, or new purchases. Runtime shutdown confirmation pending.

## Current next step: refresh evaluation, fixed RoBERTa, then context

User approved improving evaluation first, retaining RoBERTa as baseline, then trying an LLM with plot context. See `../eval-refresh/README.md`. Blinded 35-row first review batch and full 147-row dev review pack are prepared separately; original decisions and the 98 held-out rows remain intact. Human review, expanded positive coverage and work/group verification are still required. No training or new GPU run started. Future context trials begin with supplied sourced passages before implementing retrieval.

Latest user steering explicitly delegates decisions to assistant/agent. First 35 rows independently AI-reviewed under refreshed policy: 4 spoiler, 26 safe, 5 uncertain; saved in `../eval-refresh/first-batch-ai-decisions.jsonl`. Keep AI provenance; strict human/test gates unchanged. Human review is not a prerequisite for an explicitly AI-labeled exploratory dev diagnostic. Remaining 112 dev rows, five context gaps and expanded spoiler coverage still pending. No GPU run or training started.

Evaluation expansion COMPLETE: `../eval-refresh/EXPANDED_SET.md`. Current `reviewed-dev-v2.jsonl` contains 207 AI-reviewed dev examples (45 spoiler /158 safe /4 uncertain), split into refreshed original147 (5/138/4) and a separately sourced plot challenge60 (40/20/0), covering20additionalworks including5books. `source-verification.json` binds all60quotes to theirsourceURLs/hashes. `blind-inputs-v2.jsonl` is ready for text-only diagnostic loading; context inputs/inference support are still to be prepared. Report slices separately; no claims of human gold or full-review/YouTubeASR accuracy. Test98 preserved. Fixed RoBERTa configuration unchanged. No inference, training, threshold tuning or GPU provision performed in this expansion.


## October 5 supplied-context diagnostic completed

All 207 expanded AI dev examples were scored on the existing Colab T4: fixed RoBERTa, pinned Qwen with title only, and the same Qwen with supplied work context (94 examples;113 marked fallbacks). Results and verified caches are saved in `reports/context-comparison/`. Real-content spoilers caught:3/5,2/5,1/5 respectively; challenge spoilers:25/40,32/40,34/40, with false alarms:3,0,14. Keep RoBERTa baseline; this context setup does not justify a pivot. See `reports/context-comparison/RESULTS.md` and `README.md` for caveats and resume provenance. Held-out test was unchanged; no training or purchases. Runtime deletion approval is pending.
