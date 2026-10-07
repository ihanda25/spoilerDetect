# Goodreads as a possible book-data extension

Goodreads is not downloaded or used in the current pilot. Current sentence data is TV Tropes; the large review corpus is IMDb.

The official UCSD spoiler subset has over 1.3 million reviews across 25,475 books and 18,892 users. It includes book IDs and review_sentences containing sentence-level spoiler flags. Labels originate in reviewer spoiler markup, not independent expert adjudication. See [dataset details](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/reviews).

Book examples may teach narrative-reveal patterns that transfer to film and TV, but that is a hypothesis to measure on separate movie/TV data. Book metadata descriptions can be blurbs rather than complete plots; book-specific plot context still needs preparation. Group editions by work identity for held-out splits. Preserve domain-specific evaluation and compare training with versus without book examples.

The published dataset is restricted to academic, noncommercial use; production usage needs separate consideration. See [source terms](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html). No download or new training has started.
