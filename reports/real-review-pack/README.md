# Real-text human review pack

Collected 2026-10-05. **245 candidates, not 300; immutable candidate labels remain blank.** Ishaan has completed 31 human decisions, saved separately in `human-decisions-backup.jsonl`. At the user’s request, gpt-6-luna completed the remaining 214 as AI annotations in `ai-annotations.jsonl`. No real-content model evaluation has run. Import `candidates.jsonl` into the existing `tools/review_spoiler_examples.html` interface. Canonical scorer split values are **`dev` and `test`**; surfaces are **`headline`, `comment`, `review`, `youtube_transcript`**. The `youtube_transcript` enum includes the explicitly caveated subtitle-derived film excerpts below, not fetched YouTube ASR captions.

| Actual surface | Dev | Test | Total | Source coverage |
|---|---:|---:|---:|---|
| Comments | 48 | 43 | 91 | Eight Hacker News discussion threads about named films/shows |
| Headlines | 35 | 23 | 58 | 58 distinct Guardian articles across film, television, books and games |
| Reviews | 54 | 22 | 76 | 76 Guardian RSS review standfirsts, not full reviews or customer reviews |
| Transcripts | 10 | 10 | 20 | Dialogue subtitle excerpts from two Blender films available on YouTube |
| **Total** | **147** | **98** | **245** | 144 source groups; 137 connected groups; 86 declared normalized work keys |

One repeated normalized excerpt was removed from 246 collected rows. These counts describe candidates, not eligible evaluation examples, class balance or statistically independent observations. Some comments discuss production tools/technology rather than plot; some headlines concern publishing, awards or industry news. Human reviewers must assess relevance without being told an expected spoiler label.

## Files and schema

- `candidates.jsonl`: canonical complete pack, shuffled deterministically without using labels.
- `dev.jsonl`, `test.jsonl`: exact partitions of that pack; original benchmark/test datasets were not read or altered for collection.
- `blind-review.csv`: alternative review sheet omitting split and group assignments. The parent HTML interface hides split metadata and model predictions.
- `split-manifest.json`: fixed assignments and text hashes created before labeling; keep separate from human decisions.
- `summary.json`, `validation.json`: collection totals and offline integrity checks.
- `../../data/raw/product-review/collected-excerpts.json`: bounded retained excerpts and provenance, not complete fetched articles/comments.
- `../../data/raw/product-review/subtitle-excerpts.json`: manually transcribed, source-verified timed-text excerpts. Manual extraction by an assistant is **not human annotation**.
- `../../data/raw/product-review/collection-log.json`: public RSS/API request URLs, response hashes, timestamps and errors. Full HTTP responses were not retained.
- `../../scripts/prepare_real_review_pack.py`: standard-library collector and builder. `--collect` fetches public feeds; without it, the builder uses retained excerpts. It intentionally refuses to overwrite an existing review sheet. Do not rerun against human-edited artifacts; version a replacement pack instead.

Required candidate fields:

```json
{
  "id": "stable excerpt identifier",
  "text": "actual public excerpt",
  "surface": "comment | headline | review | youtube_transcript",
  "group_id": "connected source/work group identifier",
  "split": "dev | test",
  "label": null,
  "label_provenance": "human",
  "review_status": "pending_human_review",
  "source_label": null,
  "reviewer_id": null,
  "reviewed_at": null,
  "review_rationale": null
}
```

`label_provenance="human"` declares the **intended annotation source**, per the scorer contract. It does not mean a human has supplied a label. Every candidate still has `label=null` and `review_status="pending_human_review"`. There are no proposed labels, inherited corpus labels or model scores. Other fields include `source_url`, `work_title`, structured `context`, `source_group`, `work_keys`, `source_type`, extraction transformations, collection date and license metadata. Subtitle rows also retain timestamps and a separate `youtube_url`.

## Source provenance and rights metadata

**Guardian:** collected from public section RSS feeds ([film reviews](https://www.theguardian.com/film/film+tone/reviews/rss), [TV reviews](https://www.theguardian.com/tv-and-radio/tv-and-radio+tone/reviews/rss), [books reviews](https://www.theguardian.com/books/books+tone/reviews/rss), [games reviews](https://www.theguardian.com/games/games+tone/reviews/rss), plus corresponding section feeds). Review excerpts are leading standfirst text capped at 35 words; headlines at 30 words. A source article contributes to only one of these surfaces. HTML is removed and whitespace normalized; truncation is explicit. Feed notices say all rights reserved. Brief public excerpts are not an open dataset license or a guarantee of rights to redistribute a compiled corpus. Author and article URLs are retained where supplied; full review titles are not separately retained for review rows.

**Hacker News:** collected through public [Algolia HN search API](https://hn.algolia.com/api). Eight discovered story IDs are fixed in the collector: Blade Runner, Severance, Dune, The Matrix, Game of Thrones, Breaking Bad, The Good Place and Inception. Candidate selection hashes comment IDs within each bounded returned result set, takes at most 12 per thread, and limits to one comment per author across the pack. Text is capped at 35 words. Comment permalinks, discussion IDs and supplied author handles are retained; no private account data was collected. No open text license was verified. API availability does not grant a redistribution license. Query ranking and thread choice are convenience sampling, not random sampling of all public comments.

**Subtitles:** ten selected contiguous-cue excerpts each from [Sintel timed text](https://commons.wikimedia.org/w/index.php?title=TimedText:Sintel_movie_4K.webm.en.srt&oldid=175198974) and [Tears of Steel timed text](https://commons.wikimedia.org/wiki/TimedText:Tears_of_Steel_1080p.webm.en.srt). Their Commons media pages attribute the films to Blender Foundation and identify CC BY 3.0 ([Sintel](https://commons.wikimedia.org/wiki/File:Sintel_movie_4K.webm), [Tears of Steel](https://commons.wikimedia.org/wiki/File:Tears_of_Steel_in_4k_-_Official_Blender_Foundation_release.webm)). Commons timed-text footers additionally identify CC BY-SA for unstructured text; both source and license references are retained, without asserting legal equivalence. Changes are limited to cue selection/joining and whitespace normalization. Media pages identify YouTube editions [Sintel](https://www.youtube.com/watch?v=eRsGyueVLvQ) and [Tears of Steel](https://www.youtube.com/watch?v=OHOpb2fS-cM).

These are **subtitle-derived film-dialogue snippets**, not transcripts retrieved from YouTube's caption API, not creator review/recap videos, and not automatic speech recognition output. Alignment to the YouTube edition was not audio-verified. Direct Blender/Wikimedia API downloads encountered 403 responses; readable web timed-text pages supplied the excerpts. No video/audio downloads or restriction bypass occurred. Two film groups are insufficient for broad transcript generalization claims.

## Splits and independence

Source article, HN discussion thread and film/video identities are grouped before labels. Normalized review work names and known film/show names are linked by literal phrase matching across grouping context and excerpt text. Connected components are hashed with a fixed salt into roughly 70/30 dev/test allocation. The two subtitle works were explicitly assigned to opposite splits before annotation: Sintel dev, Tears of Steel test. Actual resulting split is 147/98; it was not adjusted using labels or scores.

Offline checks confirm no crossing of **declared** work keys, source groups or connected group IDs, and no duplicate normalized text. They do **not** prove exhaustive work-level independence. Review-title prefixes can be imperfect, most headlines have no identified work, aliases/franchises can be missed, and comments can mention unrecognized secondary works. `group_review_status="pending_human_verification"` records this limitation. Publisher identities appear in both splits: “source-disjoint” here means distinct articles/threads/videos, not publisher-held-out evaluation.

Before labels/model results influence choices, a curator must resolve work identities, aliases and multi-work references. Merge any cross-split conflicts into a versioned group manifest, or exclude conflicted groups from the final test. Document the change without consulting labels or predictions. Do not silently repartition a scored test set. No IMDb or TVTropes rows were included; historical text may nevertheless overlap model pretraining, so this is not proof of uncontaminated pretrained-model evaluation.

## Human review workflow and evaluation gate

1. Resolve work grouping and freeze the resulting manifest before model scoring. Adopt the parent UI policy: viewer has not started the work but knows its public premise; later outcomes, twists, hidden identities and character fates count as spoilers. Label the **displayed excerpt**, not unseen text from the linked page.
2. Import the pack into the parent review UI. A human must explicitly choose safe (`0`), spoiler (`1`) or uncertain (`null`), enter a reviewer identity and rationale, then export decisions. No assistant-generated decision is a human label. Save reviewed exports separately from this immutable candidate pack.
3. Review context/relevance and truncated excerpts. If the excerpt cannot support a reliable decision, choose uncertain; don't invent missing context or use an article's unseen spoiler content as the label. Record exclusions for irrelevant or malformed candidates rather than forcing a binary judgment. Have a second human adjudicate uncertain/disputed cases and a sample of definite cases when feasible.
4. Completed human decisions use `label_provenance="human"`, `review_status="human_reviewed"`, `reviewer_id`, `reviewed_at`, `review_rationale`. The parent UI records uncertain cases as human-reviewed with null label; those remain ineligible for binary metrics. Human-reviewed status alone must never be treated as a binary label.
5. Before scoring, validate IDs/text hashes against the manifest, binary labels, human review metadata, resolved grouping and documented exclusions. Candidate `eligible_for_evaluation=false` is intentional. The parent integration requires `group_review_status="verified"` and `eligible_for_evaluation=true` before scoring, with an explicit human grouping check in the UI. These must not be changed merely because a label is present. The pending candidate pack deliberately satisfies neither gate. Preserve excluded/uncertain counts by surface and split.
6. Choose thresholds on eligible **dev** rows only. Freeze all model/preprocessing/threshold choices, then evaluate eligible **test** rows once. Report per-surface counts, precision/recall, false positives and uncertainty; use group-level uncertainty estimates rather than treating ten snippets from one film as ten independent works.

**Current blockers:** incomplete human verification (31 human decisions backed up; 214 delegated AI annotations); unverified semantic grouping; transcript coverage limited to two clean-subtitle films; publisher/domain concentration; brief-excerpt loss of context; unknown positive counts and potentially too few examples per class after review. No real-data threshold selection or training is justified by the unlabeled pack alone. Collection is complete for this bounded pass; human review is the next required action.

## Human backup and delegated annotations

`human-decisions-backup.jsonl` preserves the 31 decisions visible in the user’s review tab (14 safe, 10 spoiler, 7 uncertain). `reviewed-with-human-backup.jsonl` retains the complete pack with those decisions. The browser’s original timestamps were not visible, so backups keep `reviewed_at=null` and record a separate `decision_captured_at`; original complete metadata can still be exported from the live tab. Group verification was unchecked on every captured decision.

The AI sidecar is explicitly `ai_reviewed`, with `label_provenance=ai` and reviewer `gpt-6-luna`. This is a first-pass annotation aid, not human gold. Do not rewrite its provenance or enable evaluation eligibility to bypass the current scorer. A combined annotation file preserves these distinctions and leaves grouping verification pending.

Completed combined annotations: `annotated-human-and-ai.jsonl` — 31 human + 214 AI; final AI counts 183 safe / 2 spoiler / 29 uncertain. See `AI_ANNOTATION_NOTES.md` for the limited quality check and correction audit. No benchmark was run.
