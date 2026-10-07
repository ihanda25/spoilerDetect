# TV Tropes dataset audit — 2026-10-01

## Decision

Useful as a small sentence-supervision research experiment; insufficient as the sole dataset for a broad spoiler-protection product. No training launched. Future fine-tuning must use a hosted GPU; do not start local training without explicit user approval. Product scope includes headlines, comments, reviews, and YouTube transcripts.

## Sources and provenance

- Repository: https://github.com/rzepinskip/spoiler-detection
- Loader: https://github.com/rzepinskip/spoiler-detection/blob/master/spoiler_detection/datasets/tvtropes_movies.py
- Original archive: https://www.umiacs.umd.edu/~jbg/downloads/spoilers.tar.gz
- Paper: https://lintool.github.io/NSF-projects/CCF-1018625/papers/2013_spoiler.pdf
- Mirror: https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_movie-train.balanced.csv (also dev1 and test).

All three mirrored splits downloaded successfully and are byte-identical to their original archive counterparts (SHA-256 recorded in audit.json). The original archive also contains dev2, which the repository loader does not use. Raw files and sampled copyrighted text remain under gitignored data/raw/tvtropes/. No external code installed or executed.

Correction: the repository calls the loader “Movies,” but the original paper describes television shows. Development samples include many TV series and reality shows. Do not present this as a verified movie-only corpus. No comprehensive work-type taxonomy was available in the CSV.

## Measured structure

| Split | Rows | Spoilers | Work pages | Median words | 95th percentile words |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 11,970 | 6,288 | 679 | 18 | 43 |
| Validation (dev1) | 1,066 | 570 | 71 | 18 | 42 |
| Test | 1,477 | 777 | 62 | 17 | 40 |

Columns: sentence, spoiler, verb, page, trope. No empty text or missing fields in these three splits. Maximum word counts: 160/165/103. These are whitespace word counts, not tokenizer counts. The labels are roughly balanced, unlike typical browsing traffic; deployment precision cannot be inferred directly from this distribution.

Original dev2: 1,748 rows from 72 additional pages, with no exact normalized page overlap with the three downloaded splits. Preserve it as a separate development resource until an evaluation plan is fixed.

## Leakage and label quality

- Zero normalized work-page overlap between train, dev1 and test. This supports unseen-page evaluation; it does not establish franchise-disjoint or alias-resolved splits.
- 32 duplicate extra rows within train and one within test after case/whitespace normalization.
- One normalized sentence shared between train/dev1 and one between train/test. The latter has conflicting labels.
- Three normalized texts have contradictory labels overall. This can reflect context dependence as well as annotation errors; don't automatically declare one label correct.
- Audit covers exact normalized duplicates, not paraphrase/near-duplicate leakage. Cross-corpus overlap with our IMDb data is not checked; IMDb IDs and TV Tropes page names need a mapping.
- Structural test inspection only: no model predictions or test-driven threshold selection performed.

Proposed cleaning for an experiment: preserve original raw splits; remove training occurrences shared with held-out splits, deduplicate training examples, quarantine contradictory training text for review. Report raw and cleaned counts. Keep held-out examples fixed and disclose their ambiguity. Do not randomly redistribute sentences across pages.

Labels originate from site spoiler markup: a sentence overlapping a marked spoiler is positive. They are not independent expert judgments for our product's spoiler policy. The paper describes filtering for verbs; nevertheless short fragments remain in the released CSV. Its treatment of common knowledge and its age make it unsuitable as a universal definition of spoilers.

Qualitative inspection: reviewed a deterministic sample of 60 rows (15 per label from train and dev1). Saw explicit plot events, pronouns without antecedents, fragments referring to omitted tropes, production trivia, and negative examples describing potentially revealing events. This was an assistant spot-check, not blinded human relabeling; no label-accuracy percentage is claimed. Row references worth human review include train 1535 (context-dependent positive), validation 824 (context-dependent positive), validation 735 (production/character-description positive), and validation 454 (potentially revealing negative). Indices are zero-based CSV data rows; text is kept in the ignored sample-review.json.

## Usage terms

The original paper states TV Tropes content is CC BY-NC-SA 3.0 (Attribution–NonCommercial–ShareAlike). The downloaded archive contains four CSVs and no standalone license text. Treat the paper's statement as provenance, not commercial clearance. Commercial deployment/training rights remain unresolved; do not assume a public download or code license licenses the underlying text for every intended use. Confirm permitted use before choosing this as a production training source. This audit does not determine legal treatment of derived model weights.

## Fit for the requested product

| Surface | What this data helps with | Missing evaluation coverage |
| --- | --- | --- |
| Headlines | Short-text spoiler clues | Headlines, clickbait, fragments and title-specific annotations |
| Comments | Sentence-level predictions | Slang, sarcasm, theories, replies and conversational context |
| Reviews | Localizing suspicious sentences | Surrounding context and reliable document-level aggregation |
| YouTube transcripts | A possible sentence/chunk scoring component | ASR errors, missing punctuation, long-range references, timestamps and video-level false-alarm rates |

Transcript support should score time-aligned segments with nearby context and return suspicious time ranges. A single high score among hundreds of segments can create excessive video-level false alarms; measure false warnings per hour and event/span recall, not just sentence F1. This dataset has no timestamps, episode-progress labels, surrounding transcript, or audiovisual annotations. It cannot validate video images/thumbnails or personalization to a viewer's progress.

## Recommended next step

1. Establish permitted use and a consistent spoiler definition (revealed fact versus speculation, premise versus later event, viewer progress).
2. Build a small human-reviewed development benchmark across all four surfaces, with work/source-disjoint holdouts and realistic negative examples. Protect future test labels from iterative tuning.
3. Benchmark the existing MiniLM without training on development sentences. Use a mapped work-disjoint subset if claiming independence from its IMDb training data.
4. If warranted, run a bounded hosted-GPU sentence-supervised pilot, starting with short sequences and a measured throughput check. Compare validation precision at matched recall against the current model. No full training run or hosted spending is authorized by this audit alone.

Reproduce structural audit: python3 scripts/audit_tvtropes.py (uses downloaded files; standard library only). Results: audit.json. Test split remains reserved for future final evaluation.
