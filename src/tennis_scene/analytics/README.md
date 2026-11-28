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
- An ingest run is unique by team, dataset, video, input SHA-256 values and the schema and definition versions. The uniqueness also covers a missing manual supplement. A clip has a database primary key `id` and the V4 string `clip_id`. Referencing tables use the database key. Event identity within a clip comes from `source_event_id` in V4 evidence. Excluded events retain the reason and any old labels in `provenance_json`. V4 subtype, certainty, side and uncertainty can be NULL when that evidence lacks them.
- A point stores score, terminal cause and forced/unforced assessment with their respective provenance. The match can reserve a NULL court profile until one is available. Geometry features reference a specific profile and preserve missing landing fields and reasons. A crossing record with a non-crossed status has NULL crossing coordinates and times. A speed window can record separate incoming and outgoing frame lists in `speed_window_frames`.
- `player_identity_maps`, `team_pair_assignments`, `serve_attempts` and `metric_snapshots` reserve Phase 2 data. A pair can reference two clip-local mappings, with its match-stable pair key, scoring team and current court side stored separately. A serve attempt has its own order, serve number, replay index and result. NULL serve number or replay index indicates missing information. The physical cause of a fault stays on the attempt.

The metric dictionary defines per-metric counting units, applicable subjects, filters, coverage and provenance. Binned results supply `filters.bins` as a mapping from dimension to bin ID and carry a `bin_version`. Candidate counts partition into resolved, unknown and explicitly excluded cases. Resolved cases form the denominator. A count uses the numerator as its value when the denominator is positive. A rate uses their ratio. A zero denominator produces `null` and `unavailable`. With a positive denominator, zero unknowns means `complete` and any unknowns mean `partial`. Evidence groups in `source_ids` can carry identifiers such as `numerator_point_ids` and `denominator_point_ids`. Point-bin deduplication uses canonical point IDs, including when multiple clips cover the same point. [Metric dictionary and record contract](metrics.py)

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
