# Review of 31 Qwen disagreements and uncertain answers

Only disagreements reviewed, with model answers visible. These are non-blind AI policy judgments, not independent human gold. Do not derive corrected overall accuracy from selectively relabeled errors. Original labels, predictions and scores unchanged.

| Review category | Count |
|---|---:|
| dataset_policy_disagreement | 13 |
| model_classification_error | 5 |
| unresolved_excerpt_or_policy_boundary | 11 |
| quote_validation_failure | 2 |

Resolved labels that differ from original markup: 14. The other 69 sentences have not been independently reviewed, so no revised full-run accuracy is reported. No API calls, retraining, or changes to historical scores.

## Concrete model issues

- Rejecting an episode-level outcome because it is not a whole-series twist (7247).
- Calling Ariel the pilot and dismissing its climactic threat (2948).
- Treating a character name, a provocative scene, or recurring recognition as consequential spoilers (3473, 8563, 7215).
- Invalid evidence quotes remain failures (7216, 7302); underlying raw labels agree with this audit, but quotes were altered.

## Recommended next step

Review all 100 examples independently with model predictions hidden, using one explicit policy for public premise, episode outcomes, romance, and minor scene details. Mark fragments with missing referents as unresolved instead of forcing a binary label. Preserve these labels separately from the source dataset. Then evaluate Qwen again against that locked review; avoid tuning the prompt to this error-only audit.

## Every reviewed example

### tvtropes:train:2913 — Firefly (2002 TV series)

When Kaylee asks how a client is chosen, Inara tries to explain that it's a process of feeling out a client through compatibility of spirit rather than actually thinking logically and carefully about it.

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: SAFE. Category: dataset_policy_disagreement.

Dialogue about client preferences, not a consequential outcome or hidden identity.

### tvtropes:train:7247 — Person of Interest (2011 TV series)

Also, Reese tries to   frame the serial rapist for cocaine possession,  but  it ends up not working because he's got good lawyers.

Original: 1; reviewed: 1; raw Qwen: SAFE; accepted Qwen: SAFE. Category: model_classification_error.

Explicitly discloses the failure of an attempted intervention against a serial rapist. Episodic outcomes count; a spoiler need not affect the entire series.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Person_of_Interest_season_1#ep4).

### tvtropes:train:3445 — Fringe (2008 TV series)

Season 3 takes place in both universes.

Original: 0; reviewed: 1; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: dataset_policy_disagreement.

The supplied public premise does not disclose parallel universes; the sentence reveals a later structural development.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Fringe_season_3).

### tvtropes:train:8425 — Sherlock (2010 TV series)

" to Sherlock while on the phone; Sherlock, wrinkling his nose, mouths back "It's fine!

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: SAFE. Category: dataset_policy_disagreement.

Truncated ordinary dialogue and a facial expression; no identifiable consequential reveal.

### tvtropes:train:8573 — Sherlock (2010 TV series)

His hand shakes when he's  not  thinking about the war!

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: SAFE. Category: dataset_policy_disagreement.

A character symptom contrasted with thinking about military service; no named later event, identity, or fate is revealed in this excerpt.

### tvtropes:train:8520 — Sherlock (2010 TV series)

Sherlock gets off on the  mystery  of the crime.

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: SAFE. Category: dataset_policy_disagreement.

General personality/interest in mysteries, consistent with the premise.

### tvtropes:train:8551 — Sherlock (2010 TV series)

Also a rare moment of vulnerability from Sherlock: His panicked "Are you all right?

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: SAFE. Category: dataset_policy_disagreement.

An emotional reaction and generic question, with no disclosed event or fate.

### tvtropes:train:3279 — Fringe (2008 TV series)

" Dressed in a hospital gown, she jumps into a cab and rants and raves to the driver about secret government conspiracies, doubles, kidnappings, and 'experiments'.

Original: 0; reviewed: None; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: unresolved_excerpt_or_policy_boundary.

Unnamed woman and missing lead-in; may be a later escape scene, but matching the surrounding plot would add information absent from the excerpt.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Olivia_(Fringe_episode)).

### tvtropes:train:3306 — Fringe (2008 TV series)

When Peter and Olivia are together and things are going good they at least move from hard liquor to wine.

Original: 1; reviewed: None; raw Qwen: SAFE; accepted Qwen: SAFE. Category: unresolved_excerpt_or_policy_boundary.

Being together can mean companionship or confirmation of romance. Need the surrounding text and a consistent relationship-spoiler boundary.

### tvtropes:train:2904 — Firefly (2002 TV series)

Mal crosses this at the battle of Serenity Valley, not even blinking when his comrade is blown away beside him when he realizes that the Independents have lost.

Original: 0; reviewed: None; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: unresolved_excerpt_or_policy_boundary.

Serenity Valley is opening backstory, not a late-series resolution. Whether this specific battle detail exceeds the allowed public premise is a policy boundary; do not force a label. Qwen’s claim about a series-finale consequence is misleading: the intended pilot aired last.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Serenity_(Firefly_episode)).

### tvtropes:train:3473 — Fringe (2008 TV series)

 Speaking of which, here's a somewhat far-fetched one from "Jacksonville"... This is one Ted Pratchett.

Original: 0; reviewed: 0; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: model_classification_error.

A character name and episode title alone do not reveal a consequential plot event. Qwen treats mere existence of an episode character as sufficient.

### tvtropes:train:2955 — Firefly (2002 TV series)

It isn't designed for this kind of thing, though  Mal's shotgun does a much better job .

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: SAFE. Category: dataset_policy_disagreement.

Vague weapon-effectiveness comparison; missing referent and no disclosed outcome.

### tvtropes:train:3367 — Fringe (2008 TV series)

Astrid:  Are you sure you don't want me to call  Olivia?

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: SAFE. Category: dataset_policy_disagreement.

Generic question about calling another character; no revealing event.

### tvtropes:train:7235 — Person of Interest (2011 TV series)

:  One episode's PoI, an investment banker, made 100 million on a short sale of Virtanen Pharmaceuticals, believing that their stock would tank when their senior management was convicted of the crimes that Reese and Finch had gotten them arrested for in an earlier episode.

Original: 0; reviewed: 1; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: dataset_policy_disagreement.

Reveals prior arrests/crimes and a specific financial result. The clause about conviction is an expectation, not confirmation; Qwen overstates that part in its explanation.

### tvtropes:train:8585 — Sherlock (2010 TV series)

Sherlock is not even above pickpocketing his own brother!

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: SAFE. Category: dataset_policy_disagreement.

Minor comic character behavior, not a consequential later outcome. The text does not reveal what was stolen or its importance.

### tvtropes:train:3254 — Fringe (2008 TV series)

She manages to find a job that she enjoys and falls in love with her partner.

Original: 0; reviewed: None; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: unresolved_excerpt_or_policy_boundary.

Unnamed character and generic romance/career summary; cannot confidently resolve the referent or premise-versus-later-outcome boundary.

### tvtropes:train:7196 — Person of Interest (2011 TV series)

Rounding out the cast of  weekly rogues  is an ensemble of recurring antagonists: Elias, an elusive mob boss and master criminal whose  conspiracies and criminal enterprises  have generated several POI numbers  not including Elias himself , "HR", a ring of  corrupt cops  within the NYPD that protect Elias and other crime syndicates, and Agent Snow, a CIA operative from Reese's past who is trying to put a bullet in him.

Original: 1; reviewed: None; raw Qwen: SAFE; accepted Qwen: SAFE. Category: unresolved_excerpt_or_policy_boundary.

An antagonist roster could be introductory description or disclose later antagonist identities. The allowed premise does not establish this; Qwen’s marketing-material justification is not evidenced.

### tvtropes:train:7216 — Person of Interest (2011 TV series)

 Root  gets the better of Reese and Finch by counting on them to do what they do best:  helping the helpless .

Original: 1; reviewed: 1; raw Qwen: SPOILER; accepted Qwen: UNCERTAIN. Category: quote_validation_failure.

Root exploits the protagonists’ rescue behavior and defeats them: an explicit later tactical outcome.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Firewall_(Person_of_Interest)).

### tvtropes:train:7302 — Person of Interest (2011 TV series)

:  In "Witness":  "You really think we'd go to this much trouble for a witness?

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: UNCERTAIN. Category: quote_validation_failure.

A rhetorical dialogue fragment does not disclose the episode’s hidden identity. Do not add that twist from plot context.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Person_of_Interest_season_1#ep7).

### tvtropes:train:7240 — Person of Interest (2011 TV series)

And yeah, the woman was part of it.

Original: 0; reviewed: None; raw Qwen: UNCERTAIN; accepted Qwen: UNCERTAIN. Category: unresolved_excerpt_or_policy_boundary.

Both woman and it are unresolved references; the excerpt cannot be confidently scored SAFE or SPOILER.

### tvtropes:train:2965 — Firefly (2002 TV series)

Jayne almost suffers this fate in "Ariel.

Original: 0; reviewed: None; raw Qwen: UNCERTAIN; accepted Qwen: UNCERTAIN. Category: unresolved_excerpt_or_policy_boundary.

This fate is missing and the quote is truncated; uncertainty is justified without reconstructing the parent paragraph.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Ariel_(Firefly_episode)).

### tvtropes:train:3510 — Fringe (2008 TV series)

Explained in the season one finale as being caused by "soft spots," which cause weird events to radiate out from them.

Original: 0; reviewed: 1; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: dataset_policy_disagreement.

Explicitly discloses an in-story explanation said to arrive in the season-one finale. Classification follows the revealing excerpt; exact source timing remains unverified.

### tvtropes:train:7266 — Person of Interest (2011 TV series)

Finch invented the concept of online social networking (and made a hefty profit in the process) specifically so that the information would be available for the machine to do so.

Original: 0; reviewed: None; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: unresolved_excerpt_or_policy_boundary.

Technical backstory can be world-building or a meaningful later revelation. The timing and consequence of this specific claim are not established by the available plot summary.

### tvtropes:train:2949 — Firefly (2002 TV series)

The test was so secret it was even a surprise to Mal, who had every intention of going throught with it.

Original: 0; reviewed: None; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: unresolved_excerpt_or_policy_boundary.

The unspecified test and its referents are missing. Qwen invents a likely loyalty/Alliance interpretation without supporting evidence.

### tvtropes:train:2894 — Firefly (2002 TV series)

The same part then shows up at the dump on Ariel, only to be tossed aside by Wash.

Original: 1; reviewed: 0; raw Qwen: SAFE; accepted Qwen: SAFE. Category: dataset_policy_disagreement.

An incidental object is discarded; no consequential outcome, hidden identity, or fate is disclosed.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Ariel_(Firefly_episode)).

### tvtropes:train:8563 — Sherlock (2010 TV series)

Done in "A Scandal in Belgravia" when Irene Adler decides to recieve Holmes and Watson totally naked.

Original: 0; reviewed: 0; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: model_classification_error.

Describes a provocative scene, not by itself a consequential outcome or twist. Qwen equates a specific scene with a spoiler without identifying a consequential reveal.

### tvtropes:train:2948 — Firefly (2002 TV series)

:  The end of "Ariel," when Mal threatens to  throw Jayne   out the airlock .

Original: 1; reviewed: 1; raw Qwen: SAFE; accepted Qwen: SAFE. Category: model_classification_error.

Explicitly reveals a later episode’s climactic threat against Jayne. Ariel is not the pilot; Qwen’s chronology and SAFE justification are wrong.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Ariel_(Firefly_episode)).

### tvtropes:train:7227 — Person of Interest (2011 TV series)

The first part definitely, the second part is still in the air but plausiblethey apparently want to shut down the evil operations of the CIA, but they are also trying to stop Reese.

Original: 0; reviewed: None; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: unresolved_excerpt_or_policy_boundary.

Missing groups and earlier clauses, plus apparently/speculative wording; cannot establish the asserted later plot reveal.

### tvtropes:train:7229 — Person of Interest (2011 TV series)

By the end of "No Good Deed", the NSA agent has acquired enough information about The Machine to get his own yellow box.

Original: 0; reviewed: 1; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: dataset_policy_disagreement.

Explicitly reveals an episode-ending acquisition of knowledge about the secret Machine.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Person_of_Interest_season_1#ep22).

### tvtropes:train:7215 — Person of Interest (2011 TV series)

:  Reese is a  Badass in a Nice Suit all Detective Carter needs to hear was that a mysterious stranger in 'a nice suit' was involved to know it was him.

Original: 0; reviewed: 0; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: model_classification_error.

Recurring recognition/clothing characterization, without an important identity twist. Qwen’s claim that Carter is introduced later is wrong; she appears in the pilot.

Sources: [plot reference 1](https://en.wikipedia.org/wiki/Person_of_Interest_season_1#ep1).

### tvtropes:train:7309 — Person of Interest (2011 TV series)

" In flashbacks, Reese and Stanton appear to get a long-distance, high-explosive version of this.

Original: 0; reviewed: None; raw Qwen: SPOILER; accepted Qwen: SPOILER. Category: unresolved_excerpt_or_policy_boundary.

This has no referent; a vague explosive flashback description does not establish a specific fate or outcome.

