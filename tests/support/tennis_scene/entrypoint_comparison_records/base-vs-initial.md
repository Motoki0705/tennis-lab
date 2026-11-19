## Independent entrypoint comparison

- Base: `e561377cec714ead088fc37569cd5fb5a893866c`
- Head: `0fdc2445964cbf6972819c0663f8df78ea143508`
- Driver: `tests/support/tennis_scene/entrypoint_comparison.py`
- Driver SHA-256: `a5bd10e3391bf35da6a7758934640bcc09dcda48cea3eda6f64ae3beb794eb3d`
- Fixed code identity: `scene-entrypoint-comparison-v1` (set before the first run and publication identity).
- Isolation: separate detached checkouts and Python processes; both stores absent before the cold run.
- The test-owned fixture path is reused only after deleting the preceding run's entire fixture and store.
- Excluded: annotation.generated_at (wall-clock time). Host paths are redacted as `<fixture>` only for display; IDs/checksums are unchanged.
- The production src fingerprint is recorded before overriding it in this fixture; production code is not patched.
- Base src tree: `14cecf580086b390d3634f8d4d67a491290d68ca`; loaded source modules verified: 427.
- Head src tree: `d310958d3fdfc0548b85075412df0f80e28e87d5`; loaded source modules verified: 430.
- Production code identities: `38b4515c4b6899914f9c74889d6ce6089742b805ea89c458277e495461bec301` / `e272d1364cf7653d25bd13b3adca3417660e4f7df03e148421cfeed5528fb2a8`.
- Runtime: `{"packages": {"numpy": "2.3.5", "pytest": "9.1.1", "torch": "2.13.0"}, "python": "3.11.15"}`; same runtime: True.

| Role | Loaded code | Git blob | SHA-256 |
|---|---|---|---|
| base | `src/tennis_scene/pipeline/runner.py` | `cee3304daa333f76491eb1ab40ddfdffac97b7a0` | `6daa43450a0ae5988c66ed95a3b39e1fcbdf8bae5b12c76fe5030b5be20744ca` |
| base | `src/tennis_scene/pipeline/orchestrator.py` | `7fffe720de7fb7b9910a6c2b117c6c36d76659ad` | `439fbeccba8f31ab96a9cc7ef1a1e363858492237d956c80aa3315bc2c812de8` |
| base | `src/tennis_scene/pipeline/definition.py` | `e664162b2209ca869c35cdcf10213b7bc439283f` | `3a3bcb6eea0b995d80a2e97af5f52e62e7daee24e7770698062ba76421b8c199` |
| base | `src/tennis_scene/generate_dataset/pseudo_annotation.py` | `b8af354bc4f03b04bfc8198c1822704d1546eab6` | `f4568e98a136973aac550d0ddfb0444ff4e7cc181cd42637c761d02b5a49f71c` |
| base | `tests/unit/tennis_scene/pipeline/test_auto_pipeline.py` | `950e0ca24ced401026cbb28f9ba56a36f1300c3d` | `493ff5f4c7f250f3cf22309289018ad4d115f1d0d9ad305cf5aa50bafed32334` |
| head | `src/tennis_scene/pipeline/runner.py` | `f30e4ee905cdce7c0a6049aa36d952bd420c1d10` | `b38cde9b555477c13f346d735df3d770602837d0ae21a74484a42f8041d312f9` |
| head | `src/tennis_scene/pipeline/orchestrator.py` | `8b7557931f91a9f87ece271a46ee9efcf9511efa` | `d4b20b8a52e2711119c91ef9bc21af81ee42f84c7de5c4f040a023c36492385c` |
| head | `src/tennis_scene/pipeline/definition.py` | `e664162b2209ca869c35cdcf10213b7bc439283f` | `3a3bcb6eea0b995d80a2e97af5f52e62e7daee24e7770698062ba76421b8c199` |
| head | `src/tennis_scene/generate_dataset/pseudo_annotation.py` | `25720533e3e55a6ec5dbf8b474eb53b40a7d4894` | `bb81a4f140956fce84c9a031ab6b45f632145848d9a63546f138064c56369c7a` |
| head | `tests/unit/tennis_scene/pipeline/test_auto_pipeline.py` | `950e0ca24ced401026cbb28f9ba56a36f1300c3d` | `493ff5f4c7f250f3cf22309289018ad4d115f1d0d9ad305cf5aa50bafed32334` |

| Output | Equal | Base digest | Head digest |
|---|---|---|---|
| scene.json/cold | True | `bd5c8c34704bdc3944563777ca7efd55cc50ce2ee0999dcb76c53d64a273b399` | `bd5c8c34704bdc3944563777ca7efd55cc50ce2ee0999dcb76c53d64a273b399` |
| scene.json/warm | True | `b500e75091b1b4fa24d7800316a575989ffe63cf55a754ca05f061f55b89db94` | `b500e75091b1b4fa24d7800316a575989ffe63cf55a754ca05f061f55b89db94` |
| scene.json/dataset | True | `c2e3f670b910d8809d980ecb560024c07e6abbd4a8eacd4894d47c60ba453e1e` | `c2e3f670b910d8809d980ecb560024c07e6abbd4a8eacd4894d47c60ba453e1e` |
| artifact_references/cold | True | `a1c5222899eea3ab68c035a43140eb5b2124589cb22a4810ed238bcbf2e063d7` | `a1c5222899eea3ab68c035a43140eb5b2124589cb22a4810ed238bcbf2e063d7` |
| artifact_references/warm | True | `a1c5222899eea3ab68c035a43140eb5b2124589cb22a4810ed238bcbf2e063d7` | `a1c5222899eea3ab68c035a43140eb5b2124589cb22a4810ed238bcbf2e063d7` |
| node_statuses/cold | True | `0fba15b07ae485a8dde583ce3c827379cf1535ac5bb9aefb1802efa732b2e790` | `0fba15b07ae485a8dde583ce3c827379cf1535ac5bb9aefb1802efa732b2e790` |
| node_statuses/warm | True | `5c10b103dba5e488706d93538ce0825a59acaa5c23d67614052e7e1c9d57f925` | `5c10b103dba5e488706d93538ce0825a59acaa5c23d67614052e7e1c9d57f925` |
| node_statuses/dataset | True | `5c10b103dba5e488706d93538ce0825a59acaa5c23d67614052e7e1c9d57f925` | `5c10b103dba5e488706d93538ce0825a59acaa5c23d67614052e7e1c9d57f925` |
| scene.metadata.json | True | `278132170ebc6fee2afacdcde91e88a16b7ca3c6ae0dfa581d952d4fd7eb2c42` | `278132170ebc6fee2afacdcde91e88a16b7ca3c6ae0dfa581d952d4fd7eb2c42` |
| annotation.json | True | `a5dd15aa79776544a65d91899ed6ca9b36c5cbaff2bedcaee9bba05b80495e12` | `a5dd15aa79776544a65d91899ed6ca9b36c5cbaff2bedcaee9bba05b80495e12` |

| Node | Cold base/head | Warm base/head | Artifact reference equal |
|---|---|---|---|
| ball_detection/cam0 | executed/executed | cached/cached | True |
| ball_detection/cam1 | executed/executed | cached/cached | True |
| ball_detection/cam2 | executed/executed | cached/cached | True |
| ball_points/cam0 | executed/executed | cached/cached | True |
| ball_points/cam1 | executed/executed | cached/cached | True |
| ball_points/cam2 | executed/executed | cached/cached | True |
| ball_triangulation | executed/executed | cached/cached | True |
| body_placement | executed/executed | cached/cached | True |
| body_view_selection | executed/executed | cached/cached | True |
| camera_alignment | executed/executed | cached/cached | True |
| court_calibration | executed/executed | cached/cached | True |
| court_detection/cam0 | executed/executed | cached/cached | True |
| court_detection/cam1 | executed/executed | cached/cached | True |
| court_detection/cam2 | executed/executed | cached/cached | True |
| court_side | executed/executed | cached/cached | True |
| gvhmr | executed/executed | cached/cached | True |
| person_detection/cam0 | executed/executed | cached/cached | True |
| person_detection/cam1 | executed/executed | cached/cached | True |
| person_detection/cam2 | executed/executed | cached/cached | True |
| person_tracking/cam0 | executed/executed | cached/cached | True |
| person_tracking/cam1 | executed/executed | cached/cached | True |
| person_tracking/cam2 | executed/executed | cached/cached | True |
| player_association | executed/executed | cached/cached | True |
| player_selection/cam0 | executed/executed | cached/cached | True |
| player_selection/cam1 | executed/executed | cached/cached | True |
| player_selection/cam2 | executed/executed | cached/cached | True |
| player_triangulation | executed/executed | cached/cached | True |
| pose_estimation/cam0 | executed/executed | cached/cached | True |
| pose_estimation/cam1 | executed/executed | cached/cached | True |
| pose_estimation/cam2 | executed/executed | cached/cached | True |
| scene_assembly | executed/executed | cached/cached | True |

Result: PASS

Reproduce:

```bash
.venv/bin/python tests/support/tennis_scene/entrypoint_comparison.py --base e561377cec714ead088fc37569cd5fb5a893866c --head 0fdc2445964cbf6972819c0663f8df78ea143508 --output-dir .comparison-output
```
