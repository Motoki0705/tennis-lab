# Tennis scene analytics

The analytics store gives ingest, hitter estimates and geometry a shared PostgreSQL schema. It also defines the versioned metric dictionary and the CLI for inspecting persisted evidence.

## Local PostgreSQL 17

Run the commands from the repository root.

```bash
uv sync --locked --no-group gpu --group cpu
docker compose up -d --wait
export DATABASE_URL='postgresql+psycopg://tennis_analytics:tennis_analytics@127.0.0.1:5432/tennis_analytics'
.venv/bin/python -m alembic upgrade head
.venv/bin/python -m src.tennis_scene.analytics db check
```

Compose accepts `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_PORT`. The development defaults are `tennis_analytics` for the first three and `5432` for the port, which is bound to loopback. Update `DATABASE_URL` when changing these values. PostgreSQL initializes the database and credentials when the named volume is first created. `docker compose down` stops the service and retains the volume.

The CLI also accepts `db upgrade [revision]` and `db downgrade [revision]`, with defaults of `head` and `base` respectively. Downgrading to `base` drops the analytics tables and their data. `db check` checks the revision and Alembic's model comparison.

## Schema and writes

Use the SQLAlchemy models and the shared `analytics_engine()` to read and write rows. The models set `updated_at` on SQLAlchemy updates. Set it explicitly for direct SQL updates. Commit a new Alembic revision for schema changes. The initial revision records its own DDL so it can replay independently of future model edits. [Models](db/models.py) · [Connection settings](db/connection.py)

- `organizations`, `teams`, `users` and `team_memberships` define the access boundary. Every domain row has a required `owner_team_id`, a `visibility` default of `team`, and timezone-aware timestamps. Domain foreign keys carry ownership and also carry clip, source or sport when both sides have that scope. The CLI is a database operator tool. Its owner filter selects rows, and runtime access policies belong to the Web API.
- An ingest run is unique by team, dataset, video, input SHA-256 values and the schema and definition versions. The uniqueness also covers a missing manual supplement. A clip has a database primary key `id` and the V4 string `clip_id`. Referencing tables use the database key.
- Event identity within a clip comes from `source_event_id` in V4 evidence. `adopted_in_v4` records whether the V4 archive adopted the annotation. `in_point` records eligibility in the point context. The first bounce after a service let can be adopted with `in_point=false`. A V4 exclusion requires `excluded_reason` and can preserve old labels in `provenance_json`. Rule and refiner events have NULL V4 adoption.
- A point stores the clip-local `point_key` and the match-level `match_point_key`. Overlapping clips of the same point share `MatchPointId(match_context_id, match_point_key)`. A NULL match point key means identity is unresolved. Terminal cause and forced/unforced assessment keep their respective provenance. V4-plus-score derivation permits `net_error` and `double_fault`. Other known causes require manual confirmation. Use `unknown` as both cause and source while unresolved.
- Geometry and crossings link to the clip that owns the underlying arcs and events. Landings have `landing_source_kind` of `observed` or `estimated`, with the bounce frame as evidence. A missing landing requires a reason and NULL landing measurements. An observed-landing metric selects `landing_source_kind='observed'`. A crossing records the court/net profile used and its source kind. A non-crossed status has NULL crossing coordinates and times. A speed window can record separate incoming and outgoing frame lists.
- `player_identity_maps`, `team_pair_assignments`, `serve_attempts` and `metric_snapshots` reserve Phase 2 data. A pair can reference two clip-local mappings, with its match-stable pair key, scoring team and current court side stored separately. A serve attempt has its own order, serve number, replay index and result. Its optional shot must belong to the same point and clip. NULL serve number or replay index indicates missing information. The physical cause of a fault stays on the attempt.

The metric dictionary defines counting units, applicable subjects, filters and coverage. Binned results supply `filters.bins` as a mapping from dimension to bin ID and carry a `bin_version`. Conditional shot error rates also require `filters.shot_role` of `serve`, `return`, `rally` or `non_serve`. The serve population uses resolved faults and legal numbered attempts. The other populations use immediate legal or net/out outcomes for the selected non-serve roles. Candidate counts partition into resolved, unknown and explicitly excluded cases. Resolved cases form the denominator. A zero denominator produces `null` and `unavailable`. With a positive denominator, zero unknowns means `complete` and any unknowns mean `partial`. [Metric dictionary and record contract](metrics.py)

Each definition lists the count relation and required numerator and denominator evidence keys. For a positive denominator, both groups in `source_ids` contain unique IDs matching their respective counts. A confirmed zero numerator has an empty numerator group. For point counts and rates, use one representative persisted point row per canonical match point. Deduplicate `MatchPointId` values by subject and bin before selecting those rows. Rally length uses contacts as numerator evidence and points as denominator evidence. `insert_metric_snapshot()`, ORM inserts and updates, and CLI reads all apply `MetricRecord` validation. [Snapshot writer](db/snapshots.py)

## Frozen V4 audit and ingest

```bash
data_root="$PWD/data"
meiji="$data_root/tennis_multiview/processed/meiji_4cam"
inputs=(--data-root "$data_root" --v4-root "$meiji/benchmarks/multiview-v4" --projects "$meiji/projects.json")

.venv/bin/python -m src.tennis_scene.analytics audit "${inputs[@]}" > v4-audit.json
.venv/bin/python -m src.tennis_scene.analytics audit "${inputs[@]}" --format csv > v4-audit.csv
.venv/bin/python -m src.tennis_scene.analytics ingest "${inputs[@]}" --owner-team 1
```

`audit` can run without a configured `DATABASE_URL`. `ingest` requires a migrated database and an existing owner team. It returns the ingest-run and match-context database IDs. Use `--manifest-sha256` and `--projects-sha256` to require specific input bytes. The audit records both digests, the SHA256SUMS digest and the manual supplement digest when supplied.

- S1 verifies the manifest, each `clip.json`, the adopted scene index, its NPZ and metadata, and the event component descriptor and sparse arrays against the release's SHA256SUMS. Camera media references are checked against the manifest, scene index and checksum inventory. `checksum_verification` reports how many files were read and hashed. Clip metadata retains release-relative paths and digests for the verified scene and event component.
- The selected Clip Studio video must contain the same clip IDs, camera origins and clip bounds as V4. Time and camera-offset comparisons use an absolute tolerance of 1 ns for floating-point representation. Confirmed points use the existing scoring rules and service order. Unconfirmed score statuses stay visible in the audit and in clip metadata. A missing, auto, unknown or multiple-point score breaks the chain. A confirmed clip after such a break needs an anchor. A `not_point` clip passes the previous score through.
- Equal source intervals or overlapping intervals with a matching certain serve identify duplicate confirmed clips. Their saved scores must agree. The first source interval supplies the shared match point key and the score advances once. A confirmed clip supplies a point association for eligible events. Point boundaries stay unresolved when the input provides only clip bounds.
- A certain terminal `net_fault` linked to a certain non-serve contact can establish `net_error` when the contact side and the confirmed winner agree. A receiver win with two explicit certain service faults can establish `double_fault`. Service lets replay the same attempt and their bounces stay outside the point. The audit reports unresolved terminal causes as `unknown` and unresolved error assessments as `undetermined`.
- The availability denominator is the full clip inventory. An observed zero for an event count is available. CSV contains the same summary, column and clip records as JSON, with `section`, `key` and `data` columns. Structured `data` cells are JSON.
- Repeating the same input hashes for the same team and definition returns the existing run. A different projects file or supplement creates a new run and preserves the earlier export.

### Manual supplement

Pass `--manual-supplement` to `audit` or `ingest`. JSON uses `{"version": 1, "points": [...]}`. CSV uses a header with the same row fields and a JSON array in each populated `serve_attempts` cell.

```json
{
  "version": 1,
  "points": [
    {
      "clip_id": "video_000/clip_000",
      "point_key": "point",
      "reviewer": "reviewer-id",
      "terminal_cause": "out_wide",
      "error_assessment": "forced",
      "note": "Checked the end of the point from two views",
      "serve_attempts": [
        {"serve_number": 1, "replay_index": 0, "result": "in"}
      ]
    }
  ]
}
```

`clip_id`, `point_key` and `reviewer` are required. Provide at least one cause, assessment or attempt. A manual terminal cause uses one of the known values in the [metric dictionary](metrics.py). Error assessments use `forced`, `unforced` or `undetermined`. Enter one review per canonical match point. A reviewed cause or assessment applies to all of its clip copies with `manual_confirmed` provenance.

List serve attempts in chronological order. Each attempt requires a `result` of `in`, `fault`, `let` or `unknown`. Optional fields are `serve_number` (1 or 2), `replay_index` (starting at 0), `source_event_id` (an in-point V4 serve from that clip) and `fault_cause` (text for a fault). A let keeps the serve number and advances the replay index. Use null or omit an optional field when its value is unresolved. Confirmed attempts are stored against the reviewed clip and point.

## CLI queries and export

```bash
.venv/bin/python -m src.tennis_scene.analytics metrics list
.venv/bin/python -m src.tennis_scene.analytics metrics export --format json
.venv/bin/python -m src.tennis_scene.analytics query ball_events --owner-team 1 --limit 20 --format csv
.venv/bin/python -m src.tennis_scene.analytics export --ingest-run 1 --owner-team 1 > run.json
.venv/bin/python -m src.tennis_scene.analytics export --match-context 1 --format csv --output-dir match-csv
```

The metric dictionary supports JSON and CSV and runs without `DATABASE_URL`. Query accepts a domain table, an optional database row `--id`, an optional numeric `--owner-team` and an optional positive `--limit`. Omit the owner filter to query all accessible rows.

A scoped export reads a single database snapshot. It includes the selected run or match, its descendant rows, the match's ingest provenance and referenced court profiles. JSON goes to stdout and CSV writes one file per domain table into a new directory. Both use database ID order per table, UTC timestamps and deterministic JSON encoding. CSV uses JSON cells for objects and arrays. A `\N` cell represents NULL, and literal text starting with a backslash gains an extra backslash.

## Tests

```bash
.venv/bin/python -m pytest -n 0 tests/unit/tennis_scene/analytics
CI_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -n 0 -m postgres tests/integration/tennis_scene/analytics
```

The unit tests run in the ordinary CPU shards. The PostgreSQL tests create and remove an isolated schema per case in the configured database, using a role with schema creation privileges. With no `DATABASE_URL` they report a skip locally. The dedicated `postgres:17` CI job sets `CI_REQUIRE_POSTGRES=1` so a missing connection configuration fails, and the aggregate `Python` check requires that job.
