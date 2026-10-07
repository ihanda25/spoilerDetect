# Supplied-context comparison

Authorized 2026-10-05: fixed RoBERTa versus Qwen alone versus Qwen with work-level context. GPU inference only on the existing Colab T4 allocation; no purchases, API calls, training or threshold tuning. The 98 held-out examples remain excluded.

Inputs are the frozen 207-example expanded AI development set. Results must separate `real_content_dev` (147, five positives) from `source_plot_challenge` (60, forty positives). All labels are assistant/agent judgments, not human gold or production accuracy. Popular works can appear in pretraining, and examples within a work are correlated.

`contexts-v1.json` contains assistant paraphrases for 23 works, including public setup and narrative progression/resolution, shared unchanged across all examples of a work. Context construction uses work-level source plots rather than candidate-specific revealing spans, decisions or reviewer rationales. Source URLs accompany each summary; Breaking Bad covers seasons one and two only. Public-premise boundaries are assistant judgments. The context hash in `context-manifest.json` freezes these inputs before inference.

Coverage: 94/207 examples have supplied context: 34/147 original-content examples and all 60 challenge examples. For the other 113 examples, the context arm explicitly reuses the title-only prediction. The common covered subset is the relevant paired comparison; all-example context scores include fallback behavior and are not evidence of successful retrieval.

Both Qwen arms get the same title and excerpt. The context arm additionally receives public premise and plot; both use the same system instructions, pinned model, greedy decoding and abstention policy. This title-aware baseline differs from the previous text-only LLM run. RoBERTa keeps its prior pinned revision, class 1, 0.5 threshold and 512-token text-only input. Qwen stays pinned to the earlier tested revision, FP16, 2048 input tokens with no silent truncation.

This is a supplied-context diagnostic, not a completed RAG system. In particular, the plot challenge and summaries use the same Wikipedia narratives; success there can show the model uses supplied information but does not establish retrieval quality or real-world spoiler detection. Prefer real-content recall, safe false alarms and abstention coverage; only five original-content positives make recall estimates unstable.

Reproduce locally without running models:

```sh
python3 scripts/build_context_pack.py
python3 -m unittest discover -s scripts -p test_compare_context_models.py
python3 scripts/build_context_colab.py
```

The generated notebook is `notebooks/Spoiler_Context_Comparison.ipynb`; its embedded input package contains project scripts, public dev excerpts, metadata, summaries and AI annotations. It contains no credentials or model weights. An appended cell in the existing authenticated notebook runs the same launcher. Per-example caches are atomically saved in `/content/spoiler-context-v1/results`; they survive ordinary browser disconnects while the runtime exists, but not runtime deletion. On completion, downloadable JSON caches and a ZIP are displayed. Save them locally before deleting the runtime. Resume requires the same fingerprint; interrupted caches can be restored into that results folder before rerunning. Annotations are read for metrics only after inference.

Launched October 5 at approximately 19:22 America/Indiana/Indianapolis in the existing Colab notebook, appended cell `DEwFCssnXu0t`. The output confirmed `GPU: Tesla T4`. Notebook: https://colab.research.google.com/drive/1O3uMsIJqaXK0-5f27l9F2vc9EYmhVDEX#scrollTo=DEwFCssnXu0t . Package fingerprint is in `package-manifest.json`. Five context integrity tests and the four existing comparison plus four eval-set checks passed. Completed in 164.194 seconds. All caches and the result ZIP have been saved locally, fingerprints verified, and the report reproduced. See `RESULTS.md`: context worsened original-content recall and caused many challenge false alarms. Runtime shutdown approval is pending.
