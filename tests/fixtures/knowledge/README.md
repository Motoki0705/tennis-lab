# Shared citation contract fixture

`citation_map/` is an isolated knowledge root's citation directory. Python tests
and the citation CLI/Web UI workers can copy it into `<KNOWLEDGE_DIR>/citation_map`
or pass it to `load_citations(path)`. These are fictional papers with reserved
`example.org` source URLs; tests need no network, model API or PDF.

Look up A–E by titles `Fixture A`–`Fixture E` or DOI `10.5555/fixture-a`–`e`.
The documents themselves are the canonical metadata; no Papers entries exist.
An independent `Fixture P` demonstrates a successful reading with reference
identification still pending, including the original lookup input and dual roles.
Its second reference is separately unverified because candidate coverage is
incomplete; both states prevent expansion and retain the original role evidence.

- A → B: comparison + foundation; A → C: foundation.
- B → D: comparison; C → D: comparison + foundation (converging branches).
- A → E: comparison + benchmark; E's full-text fetch failed.
- A/ref-4: unresolved target and unclassifiable role; A/ref-5: resolved C but
  unclassifiable role. A/ref-7 → D: component + related.
- A/ref-6 repeats B as comparison. Counts use distinct citing papers, so it does
  not increase B's comparison count. D is an unexpanded, unread leaf.

`search-fixture-comparison-depth2` reaches `[A], [B,E], [D]` and expands A/B/E.
`search-fixture-comparison-depth1` reaches `[A], [B,E]` and expands only A.
`search-fixture-mixed-depth2` reaches `[A], [B,C,E], [D]` and expands A/B/C/E.
`search-fixture-interrupted-depth2` records A/B as expanded and E as pending,
with an explicit interruption reason. Existing searches omit status/stop_reason
and therefore retain their completed-run semantics.
Only matching classified/resolved edges expand. C's edges remain available for
display but C is not expanded by the comparison-only search.
`search-fixture-pending-identification` reaches only P and expands nothing:
provider failure cannot complete a paper or turn a pending row into unresolved.

Across the whole fixture, distinct comparison citing-paper counts are B=1,
D=2, E=1. Foundation counts are B=1, C=1, D=1. Within the comparison-depth2
search's expanded sources, D's comparison count is 1 (B); C is outside that
search's expanded sources. Keep the intended aggregation scope explicit.
