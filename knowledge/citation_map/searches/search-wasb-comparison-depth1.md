---
id: search-wasb-comparison-depth1
type: citation_search
created_at: '2026-11-14T12:55:14.828683+00:00'
root_id: cite-baef928f0aefd5a6
roles:
- comparison
max_depth: 1
generations:
- depth: 0
  paper_ids:
  - cite-baef928f0aefd5a6
  expanded_ids:
  - cite-baef928f0aefd5a6
- depth: 1
  paper_ids:
  - cite-2f34272a6b8ccd3d
  - cite-63e5c3654e97f114
  - cite-aece796df0c08e14
  - cite-bee887c491b4ab99
  - cite-e660a594e07339d5
  - cite-e95495299d83d587
  expanded_ids: []
issues:
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-1
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-103
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-104
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-105
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-108
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-109
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-11
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-113
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-12
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-14
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-21
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-24
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-30
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-39
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-4
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-44
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-46
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-49
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-5
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-51
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-56
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-57
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-60
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-61
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-63
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-64
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-65
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-68
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-71
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-72
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-74
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-76
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-78
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-82
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-83
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-84
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-96
- paper_id: cite-baef928f0aefd5a6
  ref_id: ref-97
schema_version: 1
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

Attempt report (seconds are for this invocation):

```json
{
  "search_id": "search-wasb-comparison-depth1",
  "elapsed_seconds": 1.8798116829993887,
  "agy_calls": 0,
  "newly_read": 0,
  "reused": 1,
  "read_papers": 1,
  "fetch_failed": 0,
  "unresolved": 27,
  "unclassifiable": 12,
  "known_pending": 0,
  "status": "complete",
  "pending_references": 0,
  "unverified_references": 0,
  "identified_papers": [],
  "refreshed_fetch_failures": []
}
```

Authenticated resume evidence (2026-11-15):

After the shared root's reference identities were repaired by the depth-2
resume, this same search ID was explicitly executed to completion. It took
4.484 seconds including startup, reused the root reading and made zero agy or
OpenAlex requests. The completed scope has one read paper, zero full-text
failures, 113 reference rows, 27 unresolved rows (year_mismatch_or_missing 20,
no_title_match 3, author_mismatch_or_missing 2, ambiguous 2), and 12 unclassifiable
roles. Pending/unverified rows: both zero. Depth 0 is expanded; the six depth-1
targets remain unexpanded in this search. Full campaign evidence is recorded
in search-wasb-comparison-depth2. The stored generations are in the frontmatter.
