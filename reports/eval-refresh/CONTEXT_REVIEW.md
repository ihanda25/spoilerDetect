# Context adjudication audit

AI decisions, 2026-10-05. Original labels and initial AI reviews remain unchanged in their own files. `context-adjudications.json` records replacements for the versioned dev export only. No model predictions were consulted.

| Example | Earlier AI decision | New AI decision | Evidence and reason |
|---|---|---|---|
| Break Clause breakup (`real-b2194d26d6bea8bf`) | Uncertain | Safe | [Channel 4 public premise](https://www.channel4.com/programmes/break-clause) and [2026 trailer description](https://rts.org.uk/article/lara-ricote-and-samuel-bottomley-star-trailer-channel-4-sitcom-break-clause) establish the breakup/shared tenancy as the setup. |
| Sintel search subject (`real-bf4a5e2bad17538b`) | Uncertain | Safe | [Film synopsis](https://en.wikipedia.org/wiki/Sintel) establishes the search for Scales as the central premise; [timed text](https://commons.wikimedia.org/w/index.php?title=TimedText:Sintel_movie_4K.webm.en.srt&oldid=175198974) places the candidate exchange near the beginning. |
| Sintel dragon lands (`real-501b0421ca5055b1`) | Uncertain | Spoiler, minor quest progress | The same synopsis and timed text place the shaman's location/proximity revelation later in the quest. This label follows the agreed broad later-event policy; it is not the film's major identity/fate twist. |

Still unresolved from the initial batch:

- `real-181eeee6a40b0df4`: Inception's garden/reality interpretation. The [source discussion](https://news.ycombinator.com/item?id=3735039) shows an ending interpretation in the full comment, but the displayed truncated fragment does not include those concrete ending events. Do not import the unseen remainder as the candidate label. Retain uncertain.
- `real-f686ff442c178524`: Sintel's unnamed weapon history. Timed text verifies the words and early scene timing, but does not establish whether the generic violent-history statement reveals a concrete identity/backstory event. Retain uncertain rather than inventing an incident.

The broadcaster page refers to the original Comedy Blap; the current-series trailer corroborates the same premise. The Sintel synopsis is secondary evidence, not a creator-authored factual transcript; that distinction is retained. Creator pages were attempted, but some fetches failed. No restriction bypass or media download was performed.

## Remaining-development review checks

The delegated reviewer supplied distinct rationales for all 112 remaining dev examples. Parent review made these changes in the separate adjudication file:

- Sintel's lone-hunter/loneliness dialogue: safe. The agent incorrectly said the work was unidentified; the candidate already supplies the title, and [timed text](https://commons.wikimedia.org/w/index.php?title=TimedText:Sintel_movie_4K.webm.en.srt&oldid=175198974) places it in the opening exchange. No later event is revealed.
- Brothers' possible shared parentage: safe. [Pre-release first-look coverage](https://www.vanityfair.com/story/matthew-mcconaughey-and-woody-harrelson-really-are-brothers) and [showrunner interview coverage](https://www.upi.com/Entertainment_News/TV/2026/09/23/lee-eisenberg-matthew-mcconaughey-woody-harrelson-brothers-interview/4461790123222/) identify the rumor as the premise. The excerpt does not answer that question.
- Digger's drilling-caused crisis: uncertain. The [official logline](https://www.diggermovie.com/synopsis/) establishes a protagonist-caused disaster, but does not establish the specific drilling cause as public premise. Avoid assuming that extra causal detail is safe without checking trailer coverage.
- A Statement's failed conference: uncertain. The [producer synopsis](https://aloeentertainment.com/portfolio/feature-films/a-statement/) establishes the conference's task, but does not establish its failure as public premise. Historical subject matter does not automatically exempt narrative outcomes from this policy.

Result after parent adjudication: the original 147 development examples contain 5 spoilers, 138 safe, and 4 uncertain. Minor quest progress is included in positives under the agreed broad policy; it must not be conflated with major twists in subsequent reports.
