---
id: search-wasb-comparison-depth1
type: citation_search
status: interrupted
created_at: '2026-11-14T12:55:14.828683+00:00'
root_id: cite-baef928f0aefd5a6
roles:
- comparison
max_depth: 1
generations:
- depth: 0
  paper_ids:
  - cite-baef928f0aefd5a6
  expanded_ids: []
issues: []
schema_version: 1
stop_reason: 'Paper processing incomplete: cite-baef928f0aefd5a6'
---

Historical timing below predates the identification repair. The root's original
reading is retained, but its unresolved lookup results are being rechecked.
This search is now interrupted, as its frontmatter records. See
[the depth-2 repair evidence](search-wasb-comparison-depth2.md) for current counts
and the provider cooldown that prevents completion.

Attempt report (seconds are for this invocation):

```json
{
  "search_id": "search-wasb-comparison-depth1",
  "elapsed_seconds": 676.949623430999,
  "agy_calls": 1,
  "newly_read": 1,
  "reused": 0,
  "read_papers": 1,
  "fetch_failed": 0,
  "unresolved": 36,
  "unclassifiable": 12,
  "known_pending": 0,
  "status": "complete"
}
```

Integration-review correction: WASB's 77 old top-five global-search identities
are now unverified, in addition to its 36 pending rows. Its arXiv source ID and
exact HTML title were checked and the reading was preserved. This search remains
interrupted; see the depth-2 document's correction audit for current totals.

After the retry time, the same root was retried by the depth-2 search on
2026-11-15 at 09:04:40 JST. Its first reference failed with HTTP 503/429 after four
attempts. This depth-1 search therefore remains interrupted at the shared root
(depth 0 paper_ids = [cite-baef928f0aefd5a6], expanded_ids = []). It was not
separately invoked after that failure. The depth-2 document records the attempt
time, unchanged identity counts and the provider's next retry time.
