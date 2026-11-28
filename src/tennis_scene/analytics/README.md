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

## Attribute V4 shots to local players

After ingest, run the hitter rule for the selected team and run.

```bash
benchmark=data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4
.venv/bin/python -m src.tennis_scene.analytics hitter \
  --benchmark "$benchmark" --ingest-run 1 --owner-team 1
.venv/bin/python -m src.tennis_scene.analytics query shot_hitter_estimates --owner-team 1
```

The writer checks the manifest against the ingest run and verifies each clip archive and adopted shot against its V4 evidence. It saves results in one transaction and copies the event's visibility. Running again with the same evidence keeps the existing rows and timestamps. A changed rule version gets its own rows.

The rule scores each player's root and COCO17 wrists over the uncertainty window. It uses three-dimensional ball-to-landmark distance, court-side agreement, observed motion before and after contact, and mask support. Each candidate carries the ball-and-landmark valid frames and the highest-scoring contact sample. The margin compares the two highest measured scores. `local_player_index` addresses the scene array and `local_player_id` stores the corresponding value in `player_track_ids`. Both fields and the selected landmark are NULL when the rule abstains. The row retains the leading evidence and the reason.

The [rule version and thresholds](hitter.py) are tied to the [V4 visual audit and evaluation](../../../knowledge/runs/run-i2759-v4-hitter-rule-eval/README.md). Update the rule revision when changing the scoring or decision logic.

## Tests

```bash
.venv/bin/python -m pytest -n 0 tests/unit/tennis_scene/analytics
CI_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -n 0 -m postgres tests/integration/tennis_scene/analytics
```

The unit tests run in the ordinary CPU shards. The PostgreSQL tests create and remove an isolated schema per case in the configured database, using a role with schema creation privileges. With no `DATABASE_URL` they report a skip locally. The dedicated `postgres:17` CI job sets `CI_REQUIRE_POSTGRES=1` so a missing connection configuration fails, and the aggregate `Python` check requires that job.
