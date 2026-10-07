# Locked Qwen-alone versus hybrid RAG

Complete: False. Same 100 inputs and locked AI labels; 17 unresolved references excluded. Alone outputs reused, RAG outputs newly generated. No training or purchases.

| Arm | Spoilers caught | Safe falsely flagged | Abstentions | Invalid answers (all 100) |
|---|---:|---:|---:|---:|
| alone | 30/38 | 8/45 | 1 | 2 |
| rag | 1/3 | 1/4 | 1 | 0 |

Actual retrieval: title-filtered local MiniLM vector index plus BM25, sentence-only query, top3 chunks by fixed RRF. Corpus consists of cached episode summaries; incomplete coverage, particularly comic-only revelations. No label-based tuning. Passage existence does not establish passage relevance.

Accepted results require valid evidence quotes. raw-label-comparison.json separates semantic decisions from quotation failures; raw results do not certify evidence grounding. changed-decisions.json preserves passages and both responses for inspection.

These are AI development labels, not human gold. Locked labels and original baseline files remain unchanged.

## Paused hybrid RAG comparison

Paused at explicit user request with 8/100 RAG answers saved; baseline and locked labels intact. No automatic continuation. Cumulative caps: 120 request attempts and 100000 tokens; resume only on user request. [Resume instructions](RESUME.md).
