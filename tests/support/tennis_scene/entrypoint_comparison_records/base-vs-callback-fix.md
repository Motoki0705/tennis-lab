## Independent entrypoint comparison

- Base: `e561377cec714ead088fc37569cd5fb5a893866c`
- Head: `d37737a37441ac26b3a8aca31a7fd5e5311ee9ea`
- Driver: `tests/support/tennis_scene/entrypoint_comparison.py`
- Driver SHA-256: `a5bd10e3391bf35da6a7758934640bcc09dcda48cea3eda6f64ae3beb794eb3d`
- Fixed code identity: `scene-entrypoint-comparison-v1` (set before the first run and publication identity).
- Isolation: separate detached checkouts and Python processes; both stores absent before the cold run.
- The test-owned fixture path is reused only after deleting the preceding run's entire fixture and store.
- Excluded: annotation.generated_at (wall-clock time). Host paths are redacted as `<fixture>` only for display; IDs/checksums are unchanged.
- The production src fingerprint is recorded before overriding it in this fixture; production code is not patched.
- Base src tree: `14cecf580086b390d3634f8d4d67a491290d68ca`; loaded source modules verified: 427.
- Head src tree: `dd2c81423b40f5c1dc5a3d02fb2c984af87e79e7`; loaded source modules verified: 430.
- Production code identities: `38b4515c4b6899914f9c74889d6ce6089742b805ea89c458277e495461bec301` / `74de3caedcc71b38492231780d231a5b8331a692b48de1a4efcf4b5e94307290`.
- Runtime: `{"packages": {"numpy": "2.3.5", "pytest": "9.1.1", "torch": "2.13.0"}, "python": "3.11.15"}`; same runtime: True.

| Role | Loaded code | Git blob | SHA-256 |
|---|---|---|---|
| base | `src/tennis_scene/pipeline/runner.py` | `cee3304daa333f76491eb1ab40ddfdffac97b7a0` | `6daa43450a0ae5988c66ed95a3b39e1fcbdf8bae5b12c76fe5030b5be20744ca` |
| base | `src/tennis_scene/pipeline/orchestrator.py` | `7fffe720de7fb7b9910a6c2b117c6c36d76659ad` | `439fbeccba8f31ab96a9cc7ef1a1e363858492237d956c80aa3315bc2c812de8` |
| base | `src/tennis_scene/pipeline/definition.py` | `e664162b2209ca869c35cdcf10213b7bc439283f` | `3a3bcb6eea0b995d80a2e97af5f52e62e7daee24e7770698062ba76421b8c199` |
| base | `src/tennis_scene/generate_dataset/pseudo_annotation.py` | `b8af354bc4f03b04bfc8198c1822704d1546eab6` | `f4568e98a136973aac550d0ddfb0444ff4e7cc181cd42637c761d02b5a49f71c` |
| base | `tests/unit/tennis_scene/pipeline/test_auto_pipeline.py` | `950e0ca24ced401026cbb28f9ba56a36f1300c3d` | `493ff5f4c7f250f3cf22309289018ad4d115f1d0d9ad305cf5aa50bafed32334` |
| head | `src/tennis_scene/pipeline/runner.py` | `5d7c67350a976cd382ec6c92a646843df5bc2bc1` | `91081f8c7758140757b74bfddb3e564f2ad80d9cdf7562ea9aaa9d4a2353ecf0` |
| head | `src/tennis_scene/pipeline/orchestrator.py` | `8b7557931f91a9f87ece271a46ee9efcf9511efa` | `d4b20b8a52e2711119c91ef9bc21af81ee42f84c7de5c4f040a023c36492385c` |
| head | `src/tennis_scene/pipeline/definition.py` | `e664162b2209ca869c35cdcf10213b7bc439283f` | `3a3bcb6eea0b995d80a2e97af5f52e62e7daee24e7770698062ba76421b8c199` |
| head | `src/tennis_scene/generate_dataset/pseudo_annotation.py` | `25720533e3e55a6ec5dbf8b474eb53b40a7d4894` | `bb81a4f140956fce84c9a031ab6b45f632145848d9a63546f138064c56369c7a` |
| head | `tests/unit/tennis_scene/pipeline/test_auto_pipeline.py` | `950e0ca24ced401026cbb28f9ba56a36f1300c3d` | `493ff5f4c7f250f3cf22309289018ad4d115f1d0d9ad305cf5aa50bafed32334` |

| Output | Equal | Base digest | Head digest |
|---|---|---|---|
| scene.json/cold | True | `03ea9a34179d5e909ef1b23a4ca748619509ea835bd6fcd70852c114643ebe59` | `03ea9a34179d5e909ef1b23a4ca748619509ea835bd6fcd70852c114643ebe59` |
| scene.json/warm | True | `ab3343315065afa269ffbab9a3aba85bff35e56a310f63264440903dcbcafab8` | `ab3343315065afa269ffbab9a3aba85bff35e56a310f63264440903dcbcafab8` |
| scene.json/dataset | True | `980f48c10e6d2f63cbdaf235d5873416990a388db139e25b1c2498eca1b0d627` | `980f48c10e6d2f63cbdaf235d5873416990a388db139e25b1c2498eca1b0d627` |
| artifact_references/cold | True | `04eb13523a162a710ebb20329de3fb159271817dfb812eee8096ffdaabae1e44` | `04eb13523a162a710ebb20329de3fb159271817dfb812eee8096ffdaabae1e44` |
| artifact_references/warm | True | `04eb13523a162a710ebb20329de3fb159271817dfb812eee8096ffdaabae1e44` | `04eb13523a162a710ebb20329de3fb159271817dfb812eee8096ffdaabae1e44` |
| node_statuses/cold | True | `0fba15b07ae485a8dde583ce3c827379cf1535ac5bb9aefb1802efa732b2e790` | `0fba15b07ae485a8dde583ce3c827379cf1535ac5bb9aefb1802efa732b2e790` |
| node_statuses/warm | True | `5c10b103dba5e488706d93538ce0825a59acaa5c23d67614052e7e1c9d57f925` | `5c10b103dba5e488706d93538ce0825a59acaa5c23d67614052e7e1c9d57f925` |
| node_statuses/dataset | True | `5c10b103dba5e488706d93538ce0825a59acaa5c23d67614052e7e1c9d57f925` | `5c10b103dba5e488706d93538ce0825a59acaa5c23d67614052e7e1c9d57f925` |
| scene.metadata.json | True | `278132170ebc6fee2afacdcde91e88a16b7ca3c6ae0dfa581d952d4fd7eb2c42` | `278132170ebc6fee2afacdcde91e88a16b7ca3c6ae0dfa581d952d4fd7eb2c42` |
| annotation.json | True | `32da09bef4a6d35e8a3835fcecc925ed7d7bdc415ed6525aa01dd0c8c6044a0e` | `32da09bef4a6d35e8a3835fcecc925ed7d7bdc415ed6525aa01dd0c8c6044a0e` |

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
.venv/bin/python tests/support/tennis_scene/entrypoint_comparison.py --base e561377cec714ead088fc37569cd5fb5a893866c --head d37737a37441ac26b3a8aca31a7fd5e5311ee9ea --output-dir .comparison-output
```
