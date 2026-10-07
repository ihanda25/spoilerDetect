# Colab inference benchmark

`Spoiler_RoBERTa_Benchmark.ipynb` is self-contained: it embeds the 64 synthetic development examples and saved epoch-2 MiniLM predictions. It does not upload weights, train, mount Drive, or provision paid compute.

Colab copy created 2026-10-05:
https://colab.research.google.com/drive/1L5WnjeGaDgUCpclJ8Hh7FnY6-zo8adW4

Choose T4 GPU and Run all. The notebook downloads pinned Zritze RoBERTa safetensors, checks CUDA and label metadata, runs inference, prints overall/per-surface metrics, and downloads a ZIP of results, predictions, and errors. Preserve the ZIP before disconnecting. Access to the Colab copy follows the owner's existing sharing settings.

These are provisional synthetic labels, not a human-reviewed accuracy test. Twelve uncertain examples are excluded. RoBERTa uses 0.5; MiniLM uses its previously validation-selected threshold. Do not tune thresholds on these examples or infer architecture superiority from this small diagnostic. Both systems truncate long inputs.

## Real-content follow-up

`Spoiler_Real_Content_Evaluation.ipynb` is built by `scripts/build_product_colab.py`.
It imports a complete human-reviewed dev/test JSONL, runs RoBERTa truncate/window
on development, freezes the winning configuration and threshold, and provides an
explicit final test cell. Human review and verified grouping are required. It has
not run on real content because human review metadata and group verification remain incomplete.
See `reports/product-evaluation/NEXT_RUN.md` for resuming and MiniLM comparison.

## Three-model baseline (2026-10-05)

`Spoiler_Three_Model_Comparison.ipynb` compares MiniLM epoch2, Zritze RoBERTa and Qwen2.5-3B-Instruct on 147 development excerpts, text only. Upload ignored `models/three-model-comparison-inputs.zip` when prompted. The LLM runs on T4 without API keys or RAG; approximately 6 GB weights. No training. Human/AI label agreement is separate; test remains excluded. See `../reports/three-model-comparison/README.md` for caveats and the uploaded notebook URL.
