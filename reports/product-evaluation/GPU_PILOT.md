# Hosted-GPU pilot proposal — not launched

User requested deeper investigation of RoBERTa and DistilBERT training recipes on 2026-10-02. This is preparation only. No GPU purchase, rental, paid API use or training has been authorized/launched. Tell the user before heavy fine-tuning so they can arrange the GPU.

## Hardware

A single NVIDIA T4 (16 GB) is a reasonable initial calibration target for these encoder classifiers with small microbatches, mixed precision and bounded sequence lengths. This is an engineering proposal, not a tested fit/runtime guarantee. Start with DistilBERT to validate the pipeline, then RoBERTa-base if worthwhile. Use FP16 mixed precision on T4; detect actual device capability rather than assume the available GPU. Record model/tokenizer revisions, software versions and GPU model.

Colab can provide free GPU access, but neither GPU type nor uninterrupted runtime is guaranteed. Its FAQ describes up to 12 hours for free notebooks subject to availability and usage patterns. A notebook with periodic persisted resumable checkpoints is appropriate; do not promise the whole experiment fits one free session. Keep active data/token caches on runtime disk, periodically save model/optimizer/scheduler/RNG/progress to persistent storage and verify restoring a checkpoint before a long run. Do not use browser keepalive or remote-SSH workarounds for free-tier limits.

Sources checked 2026-10-02:
- https://research.google.com/colaboratory/faq.html
- https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf
- https://huggingface.co/docs/transformers/en/perf_train_gpu_one

## Separate two questions

1. Can we reproduce a publisher's result? Use the exact published split/labels and reported hyperparameters, with missing choices explicitly documented. Matching a headline number on another distribution is not a reproduction.
2. Does the model improve our product? Use the same independent, human-reviewed content and operating-point selection protocol for every candidate. Unknown overlap between public IMDb models and our historical test split prevents calling that an independent comparison. Do not discard the existing baseline.

Benchmark the existing downloadable checkpoints before spending on replication. An inference comparison may eliminate the need for retraining. Any uncertain label mapping must be resolved from primary sources before scoring.

## Proposed first training session after explicit go-ahead

- Use a fixed, documented development dataset/subset; avoid immediately repeating 444K reviews over many epochs.
- Tokenize with each model's own tokenizer. Report truncation; don't reuse MiniLM token IDs for RoBERTa/DistilBERT.
- Start microbatch 4; increase only after measuring headroom. If targeting effective batch 16, use accumulation 4. These are pilot choices, not asserted publisher settings.
- Choose max length explicitly (e.g. 256 for a cheap pilot); shortening relative to publisher training makes it an adaptation, not exact reproduction. Include long examples in timing and memory measurements.
- Time at least 100 representative optimizer steps after warmup, synchronize CUDA when timing, record peak allocated/reserved memory. Warmup or a sample of only short inputs is not a reliable full-run estimate.
- Print projected epoch time including a separate evaluation estimate. Stop before the long phase and present the measured budget to the user.
- When approved, run one epoch, persist checkpoints every fixed number of optimizer steps and at epoch end, validate, then stop. No automatic second/third epoch.
- Compare validation average precision, F1 after validation-only threshold selection, precision at matched recall, false alarms per surface, inference latency and truncation failures. Keep uncertain labels out of binary metrics and count them separately.

Exact publisher hyperparameters and unresolved details belong in the model-research report. No notebook has been connected to the user's account or GPU provisioned as part of this proposal.
