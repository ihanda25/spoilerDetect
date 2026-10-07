# Colab RoBERTa synthetic diagnostic — 2026-10-05

Completed on Tesla T4 in the saved Colab notebook:
https://colab.research.google.com/drive/1L5WnjeGaDgUCpclJ8Hh7FnY6-zo8adW4

Observed directly in notebook output. All 64 examples processed; 52 provisional binary labels scored, 12 uncertain labels excluded. No human review: these are synthetic diagnostic results, NOT production accuracy.

| System | Threshold | Accuracy | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---|
| Zritze RoBERTa LR1 | .5 | .75 | .652174 | .75 | .697674 | 24/8/5/15 |
| Saved epoch-2 MiniLM | .21579217910766602 | .75 | .769231 | .50 | .606061 | 29/3/10/10 |

RoBERTa caught five more proposed spoilers but flagged five more proposed safe examples. All four late-reveal reviews were missed; these were truncated. All four spoiler-free craft comments and four spoken-craft transcript examples were false positives. Thresholds were not tuned on these examples. Different configured thresholds preclude attributing differences solely to architecture.

RoBERTa forward evaluation with tokenization: 2.071222 seconds after loading/warmup; setup cell 38 seconds, loading/evaluation cell 41 seconds. Download/setup time is separate from the 2.07-second evaluation measurement.

Model: Zritze/imdb-spoiler-robertaOrigDatasetLR1, revision 56fee120f8495ccfc5001e3fbd1478656d17001a. Published map 0 NON-SPOILER / 1 SPOILER. Safetensors only, no remote code. Torch 2.11.0+cu130; Transformers 4.57.1. Batch 4, max 512 tokens, FP32, inference only.

The notebook generated results.json, predictions.jsonl, errors.json and requested download of spoiler-benchmark-results.zip. Local ZIP receipt has not been verified; preserve it before runtime deletion. This report records the visible summary only.

## Overlapping-window follow-up

Same model and fixed .5 cutoff, 512-token windows with 128-token overlap; max score over all windows. Evaluated 88 windows for the same 64 synthetic examples, 52 scored. CUDA-synchronized evaluation including tokenization: 1.929 seconds; this small-run timing is not a speed comparison.

Window results: accuracy .826923, precision .703704, recall .95, F1 .808511; TN24/FP8/FN1/TP19. All four late-reveal examples were recovered (scores .7716, .5574, .7638, .7968 in original work order). False positives remained eight on this diagnostic. The unresolved positive is the Last Orchard early-reveal review. These provisional synthetic findings justify real-content evaluation, not a deployment-quality claim. No threshold was selected on this diagnostic set.
