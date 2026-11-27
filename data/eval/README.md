# data/eval

`queries.jsonl` - 20 hand-labelled retrieval queries over the committed 200-report
sample (`data/sample/asrs_sample.csv`).

Each line has:

| field | meaning |
|---|---|
| `id` | query id |
| `theme` | the incident type the query describes |
| `query` | a newly written narrative in a reporter's voice (not copied from any report) |
| `relevant` | ACNs in the sample judged relevant |
| `relevance_criterion` | the rule used to judge relevance for this query |
| `review_status` | `draft - pending owner review` until the labels are checked |

**How the labels were made.** Every one of the 200 sample reports was judged against
each query's criterion (exhaustive judging, not pooling from system output), using the
synopsis and keyword-matched excerpts of the narrative. Judgments were made before
running any retrieval system on these queries. They are drafts: please review them,
edit `relevant` where you disagree, and change `review_status` to `reviewed`.
Re-run `python -m asrs_rag --index-dir data/index_sample evaluate labelled` afterwards.
