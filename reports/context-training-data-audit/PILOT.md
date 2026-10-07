# Sentence-plus-context preparation pilot — October 6, 2026

Completed preparation, not training. User authorized a mapping audit and small paired pilot. The existing RoBERTa baseline is unchanged. Gemini remains stopped. No hosted GPU, API inference or paid services used.

## What was built

Inventoried all 884 TV Tropes work pages including reserved dev2. Seven selected series identities were checked against public Wikipedia source pages and the names/episode references in sampled train/dev sentences. No TV Tropes-to-IMDb-ID join was invented. Existing IMDb metadata lacks canonical titles, and none of our existing 23 diagnostic context titles match these page names after normalization. All other 877 page identities remain unresolved; seven mappings are a measured lower bound, not the maximum possible coverage.

| Split | Mapped candidate sentences | Pilot sentences | Source positives | Series |
| --- | ---: | ---: | ---: | ---: |
| Train | 744 | 200 | 100 | Fringe, Sherlock, Firefly, Person of Interest |
| Validation/development | 198 | 60 | 34 | Homeland, Nikita, British Life on Mars |

942 candidate pairs preserve original labels and row IDs. Draft contexts are short assistant paraphrases of publicly sourced series descriptions. The 260-row pilot samples up to 50 train and 20 dev rows per work, targeting balanced labels where available. It is not representative browsing prevalence. Test and reserved dev2 were excluded from content mapping, context building and pilot selection. Their normalized texts are used only to remove exact train leakage. No cleaning exclusions occurred among these seven selected works. Work pages and context-source URLs are disjoint across pilot splits, as are normalized training texts from all held-out splits. Source CSV hashes recorded in the manifest remain unchanged.

## Preparation quality result

Reviewed a fixed label-independent sample of 8 candidate rows per mapped work (56 total), using draft context, original excerpt and source label. This is an assistant first pass, not a blinded human accuracy estimate:

- 23: required event missing from current context.
- 17: fragments, omitted trope or unresolved references.
- 4: context supports only part of the proposition.
- 1: revealed proposition supported by the current context.
- 6: non-plot commentary/production reference or generic insult.
- 3: references outside the mapped series scope, such as an American remake or film continuation.
- 2: scope/identity unresolved at sentence level.

These counts assess our short draft contexts and sampled excerpt usability, not inherent corpus quality or model accuracy. In particular Homeland and Sherlock drafts are largely premise-level, which predictably misses specific episode facts. Many originals refer to named episodes; some source SAFE labels describe substantial plot events, while some source SPOILER labels are generic fragments. No original labels were silently changed. All pilot records explicitly have training_allowed=false and human_verified=false; this is a preparation pack, not a ready-to-train dataset.

## Decision

Do not spend GPU time on these drafts yet. Build episode-level context for the selected seven series, resolve cross-work/remake references, and review the sampled labels against the intended spoiler policy. Increase grounded coverage and work diversity before a meaningful sentence-versus-paired training comparison. This is a refinement within the authorized data-preparation step, not an abandonment of context-conditioned RoBERTa. The episode-context-queue.json contains row IDs and next actions. No automatic training or spending.

## Files and reproduction

Run `.venv/bin/python scripts/build_sentence_context_pilot.py`. It reads frozen pilot-contexts.json and pilot-quality-judgments.json and writes ignored data/processed/sentence-context-pilot/{candidate-pairs,train,validation,quality-review}.jsonl. The work inventory, hashes, coverage and quality counts are saved in work-inventory.jsonl and pilot-manifest.json. Raw copyrighted excerpts remain in gitignored data/processed; the report only references row IDs. Context source URLs and sections are included in pilot-contexts.json. TV Tropes source usage caveats remain in ../tvtropes-audit/AUDIT.md.

Sources checked October 6: [Fringe](https://en.wikipedia.org/wiki/Fringe_(TV_series)), [Sherlock](https://en.wikipedia.org/wiki/Sherlock_(TV_series)), [Firefly](https://en.wikipedia.org/wiki/Firefly_(TV_series)), [Person of Interest](https://en.wikipedia.org/wiki/Person_of_Interest_(TV_series)), [Homeland](https://en.wikipedia.org/wiki/Homeland_(TV_series)), [Nikita](https://en.wikipedia.org/wiki/Nikita_(TV_series)), [Life on Mars](https://en.wikipedia.org/wiki/Life_on_Mars_(British_TV_series)). Identity/adequacy judgments are ours; public sources do not verify the training labels.

October 6 episode follow-up: episode passages added to 258/260 frozen pilot rows; event coverage remains unverified. See [EPISODE_CONTEXT.md](EPISODE_CONTEXT.md). Goodreads is a possible book-data extension, not currently loaded; see [GOODREADS.md](GOODREADS.md). No training started.
