---
id: search-wasb-comparison-depth2
type: citation_search
status: interrupted
created_at: '2026-11-14T13:09:45.637972+00:00'
root_id: cite-baef928f0aefd5a6
roles:
- comparison
max_depth: 2
generations:
- depth: 0
  paper_ids:
  - cite-baef928f0aefd5a6
  expanded_ids: []
issues: []
schema_version: 1
stop_reason: 'Paper processing incomplete: cite-baef928f0aefd5a6'
---

Historical reports below preserve the original measurements. Their earlier
`complete` status was invalid: transport failures had been finalized as
unresolved references. Current frontmatter and the repair evidence at the end
supersede that completion claim; the original readings are retained.

Attempt report (seconds are for this invocation):

```json
{
  "search_id": "search-wasb-comparison-depth2",
  "elapsed_seconds": 424.8445194229971,
  "agy_calls": 1,
  "newly_read": 1,
  "reused": 1,
  "read_papers": 1,
  "fetch_failed": 0,
  "unresolved": 36,
  "unclassifiable": 12,
  "known_pending": 6,
  "status": "interrupted"
}
```

Attempt report (seconds are for this invocation):

```json
{
  "search_id": "search-wasb-comparison-depth2",
  "elapsed_seconds": 1019.574230981998,
  "agy_calls": 3,
  "newly_read": 3,
  "reused": 2,
  "read_papers": 5,
  "fetch_failed": 2,
  "unresolved": 133,
  "unclassifiable": 38,
  "known_pending": 0,
  "status": "complete"
}
```

Attempt report (seconds are for this invocation):

```json
{
  "search_id": "search-wasb-comparison-depth2",
  "elapsed_seconds": 3.7204602910023823,
  "agy_calls": 0,
  "newly_read": 0,
  "reused": 7,
  "read_papers": 5,
  "fetch_failed": 2,
  "unresolved": 133,
  "unclassifiable": 38,
  "known_pending": 0,
  "status": "complete"
}
```

Execution evidence:

- The depth-1 measurement was recorded in PR #2044 before depth 2 began.
- This search was deliberately interrupted after the MonoTrack paper checkpoint,
  then resumed with the same root, roles, depth, and search ID. Both the WASB and
  MonoTrack documents remained byte-identical through that resume.
- A final cached rerun made zero model calls and left all 115 paper documents
  byte-identical. It reused seven expanded records (five read papers and two
  full-text failures). The final frontier contains one unread depth-2 leaf.
- All five model calls across the depth-1 and depth-2 campaign used agy 1.5.9
  with the observed label Gemini 4 Hydrogen (High). No model fallback was used.
- The old unresolved count included 112 OpenAlex HTTP 429 failures. Those were
  implementation errors, corrected below; they are not evidence of absent or
  ambiguous literature identities.

Attempt report (seconds are for this invocation):

```json
{
  "search_id": "search-wasb-comparison-depth2",
  "elapsed_seconds": 127.64005422800255,
  "agy_calls": 0,
  "newly_read": 0,
  "reused": 7,
  "read_papers": 5,
  "fetch_failed": 2,
  "unresolved": 38,
  "unclassifiable": 38,
  "known_pending": 1,
  "status": "interrupted",
  "pending_references": 36,
  "identified_papers": [
    "cite-2f34272a6b8ccd3d",
    "cite-aece796df0c08e14",
    "cite-e660a594e07339d5",
    "cite-e95495299d83d587"
  ],
  "refreshed_fetch_failures": [
    "cite-63e5c3654e97f114",
    "cite-bee887c491b4ab99"
  ]
}
```

Attempt report (seconds are for this invocation):

```json
{
  "search_id": "search-wasb-comparison-depth2",
  "elapsed_seconds": 2.8854801430061343,
  "agy_calls": 0,
  "newly_read": 0,
  "reused": 7,
  "read_papers": 5,
  "fetch_failed": 2,
  "unresolved": 38,
  "unclassifiable": 38,
  "known_pending": 1,
  "status": "interrupted",
  "pending_references": 36,
  "identified_papers": [],
  "refreshed_fetch_failures": []
}
```

Repair evidence (2026-11-14 UTC):

| Measure | Before repair | After repair attempts |
|---|---:|---:|
| Unique completed readings | 5 | 5, all reused |
| Full-text failures | 2 | 2 |
| Bibliography rows | 255 | 255 |
| Resolved reference rows | 122 | 181 |
| Unresolved reference rows | 133, including 112 transport failures | 38, after successful candidate queries |
| Pending identity rows | 0 | 36 in WASB |
| Unclassifiable role rows | 38 | 38 |
| Paper documents | 115 | 157 |

- Both repair attempts used the same root, comparison role, maximum depth 2 and
  search ID. No agy call or additional paper reading occurred. The five original
  titles, summaries, bibliography strings, roles and evidence locations were
  checked against their pre-repair values and preserved.
- The four depth-1 sources now use their source OpenAlex referenced works:
  MonoTrack unresolved 31→13, DeepBall 24→12, BallSeg 10→2, FootAndBall 32→11.
  There are 59 newly resolved rows. Remaining unresolved reasons are year
  mismatch/missing (27), no title match (9), author mismatch/missing (2).
- Of the old 112 transport failures, 59 now resolve, 38 are confirmed unmatched,
  and 15 remain pending. WASB's other 21 old unmatched rows were also scheduled
  for rechecking under the new matching rules, giving 36 pending rows total.
- BallTrack and TrackNetV2 were retried. Their documents record the DOI candidate
  attempted and OpenAlex best OA/other location diagnostics; no public PDF URL
  was provided. They remain explicit full-text failures.
- The known resolved comparison edges imply four distinct depth-2 candidates:
  Ball localization for RoboCup soccer, CNN ball detection in tennis, Mask R-CNN,
  and the Mask R-CNN Keras/TensorFlow implementation. These are cached edge facts;
  depth-2 generations are not finalized while the root is incomplete.
- A real OpenAlex query returned HTTP 429 with Retry-After 33838 seconds, longer
  than the six-hour campaign budget. WASB's indexed source has no referenced-work
  candidates, so its remaining rows need global search. The server cooldown is
  retained privately across restarts; no early query retry was sent. The two
  repair commands exited 1, and both affected searches are interrupted.
- Repair attempt times inside Explorer were 127.640 and 2.885 seconds, totaling
  130.526 seconds. Including Python/CLI startup, they took 130.471 and 5.210
  seconds, totaling 135.681 seconds. The second attempt kept all 157 paper
  documents byte-identical before this explanatory annotation was added.
- Remaining work is identity lookup for 36 rows and rebuilding both searches.
  No comparison row is pending, so the current classifications require no new
  readings for depth 2. Once query service is available, allow roughly 1–3
  minutes for those lookups at the configured one-second pacing; this estimate
  excludes provider waiting and lacks a successful-search timing measurement.
  Resume with the same CLI arguments and private run directory. The recorded
  cooldown conservatively ends after 2026-11-15 00:00:58 UTC.

Integration-review correction (2026-11-14 UTC):

The preceding 181 resolved / 36 pending counts predate complete-search
verification. They are superseded by this audit, not by a new exploration run.
The five saved reading sources were fetched as HTML and checked against their
stored arXiv IDs and exact titles. All five match; no PDF-anchor fallback was
used by these readings, and no reading or classification was replaced.

The old top-five global search did not establish uniqueness. Its 78 formerly
resolved rows were reverted to unverified: WASB 77, MonoTrack 1. Their prior
target IDs remain in the reasons, and original lookup inputs were recovered from
the matching private readings. Reference strings, roles, decision states,
evidence locations, summaries and external IDs were preserved. The other
103 resolved rows use explicit identifiers (53) or the source's complete
referenced-work candidate set (50), and were retained. Both searches were rebuilt.

| Measure | Before review correction | After offline correction |
|---|---:|---:|
| Completed readings / full-text failures | 5 / 2 | 5 / 2 |
| Resolved / confirmed unresolved rows | 181 / 38 | 103 / 38 |
| Pending / uniqueness-unverified rows | 36 / 0 | 36 / 78 |
| Unclassifiable role rows | 38 | 38 |
| Bibliography rows / paper documents | 255 / 157 | 255 / 157 |

Confirmed unresolved reasons remain year mismatch/missing (27), no title match
(9), author mismatch/missing (2). The correction made zero OpenAlex requests and
zero agy calls. It is an offline cache invalidation, not a timed WASB rerun.
The provider's retry time is still in the future for this correction, so the
exploration was not resumed. There are now 114 identity rows to recheck rather
than 36. New comparison identities may require further readings; the required
count is unknown until those rows are reidentified. The five existing readings
remain reusable. Unverified and pending rows prevent completed expansion.
