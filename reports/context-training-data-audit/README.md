# Context-conditioned RoBERTa data audit — October 6, 2026

Measured from the local IMDb archive, prepared split arrays, TV Tropes CSVs and original archive. No training, downloads, API inference, or GPU spending. Reproduce corpus counts with `.venv/bin/python scripts/audit_context_training_data.py`; audit.json also includes the saved supplied-context manifest.

IMDb contains 573,913 review-level labels (150,924 positive). 573,906 reviews join to a nonempty plot summary; seven reviews from two IDs lack metadata. All 1,572 metadata rows have nonempty summary strings. 1,339 metadata rows have nonempty longer synopses, covering 538,828 raw reviews. Presence does not establish factual quality or ending coverage. Median summary 96 whitespace words, synopsis 1,260 words; synopsis p90 3,386.8 words. Relevant-passage selection is necessary to fit a short encoder input when using synopses.

Existing duplicate/empty-filtered, movie-disjoint splits:

| Split | Reviews | With summary | With synopsis | Unique movie IDs |
| --- | ---: | ---: | ---: | ---: |
| Train | 444,644 | 444,637 | 419,336 | 1,244 |
| Validation | 63,836 | 63,836 | 59,793 | 164 |
| Test | 64,405 | 64,405 | 58,703 | 164 |

Zero movie-ID overlap across these splits. These are reviews, not labeled sentences: splitting a positive review and copying its label to all sentences would create false positive supervision. A held-out IMDb split is independent of our local MiniLM training, but external RoBERTa checkpoint training provenance is incomplete; do not claim its independence from this corpus.

TV Tropes: train 11,970 sentences / 6,288 positives / 679 work pages; dev1 1,066 / 570 / 71; test 1,477 / 777 / 62. Total 14,513 sentence examples. Original archive has a separate reserved dev2 with 1,748 sentences / 938 positives / 72 pages. Rows provide work-page names but no plot summaries or canonical IMDb IDs. No existing audited sentence-to-plot training join is ready. Many works are TV shows; map canonical works and seasons/episodes before choosing context. Existing audit notes duplicates, conflicting labels, source markup rather than expert labels, and unresolved production usage terms. See ../tvtropes-audit/AUDIT.md. Preserve held-out pages and reserve dev2.

Current paired development diagnostic covers only 94 of 207 examples using 23 assistant-written work contexts. It is AI-reviewed development data, not a training corpus or independent gold test; 60 plot-challenge cases share source plots with the contexts. Preserve it for diagnostics. There is no substantial verified book or YouTube transcript training corpus in this inventory.

Earlier MiniLM v3 already trained review-plus-summary pairs for one epoch. It improved validation recall (0.487 to 0.540 versus v2 epoch 1) but did not improve F1 (0.5602 to 0.5558). This does not settle the sentence-plus-relevant-context hypothesis; its review component was truncated more aggressively. Source: ../full-v2-plot/PLAN.md.

Recommended next step: audit TV Tropes work-page mapping and measure attainable plot-context coverage before promising sentence training volume. Prepare a small paired sentence development pilot and a matched sentence-only baseline with identical page-disjoint splits. Use richer synopses/relevant passages, assess summary coverage, and human-check representative labels. Keep the existing RoBERTa baseline unchanged. Decide GPU training only after paired data quality/coverage and a bounded run estimate are available. No training authorized by this audit alone.

October 6 follow-up: completed seven-work mapping and a 260-row paired preparation pilot. 942 candidate sentences mapped; draft contexts inadequate for training. See PILOT.md and pilot-manifest.json for measured coverage, quality review and episode-context follow-up. No training started.

October 6 episode follow-up: episode passages added to 258/260 frozen pilot rows; event coverage remains unverified. See [EPISODE_CONTEXT.md](EPISODE_CONTEXT.md). Goodreads is a possible book-data extension, not currently loaded; see [GOODREADS.md](GOODREADS.md). No training started.
