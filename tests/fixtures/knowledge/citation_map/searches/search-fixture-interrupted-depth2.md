---
schema_version: 1
type: citation_search
id: search-fixture-interrupted-depth2
created_at: '2026-11-14T00:00:00Z'
root_id: cite-78f78b45b14c940f
roles: [comparison]
max_depth: 2
status: interrupted
stop_reason: 'agy:cli_error (fictional quota interruption before E)'
generations:
- depth: 0
  paper_ids: [cite-78f78b45b14c940f]
  expanded_ids: [cite-78f78b45b14c940f]
- depth: 1
  paper_ids: [cite-15ff6ca63fce212e, cite-9278de4b09038c2d]
  expanded_ids: [cite-15ff6ca63fce212e]
issues:
- paper_id: cite-78f78b45b14c940f
  ref_id: ref-4
- paper_id: cite-78f78b45b14c940f
  ref_id: ref-5
---

Fictional interrupted run. A and B are complete; E was reached but not expanded
by this run. B's comparison edge to D is retained in B's document. Resume uses
those cached decisions, attempts E, and then records the depth-2 leaf D.
