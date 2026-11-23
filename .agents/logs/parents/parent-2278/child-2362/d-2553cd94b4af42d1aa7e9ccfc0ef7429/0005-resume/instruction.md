## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2379,
  "head": "a9d2545663ea5e412d1c8b41b5f4caf75f0c2f65",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "System dependency tests",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244045",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:35:57.1898692Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:35:57.1900115Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:35:57.1901533Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:35:57.1902938Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:35:57.1903859Z ============ 17 passed, 16 warnings, 10 errors in 101.33s (0:01:41) ============\n2026-11-22T10:35:58.3892447Z - JUnit failure count differs from phase reports\n2026-11-22T10:35:58.3902584Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:35:58.4061296Z ##[error]Process completed with exit code 1.\n2026-11-22T10:35:58.9481505Z - JUnit failure count differs from phase reports\n2026-11-22T10:35:58.9487174Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:35:58.9632793Z ##[error]Process completed with exit code 2.\n2026-11-22T10:36:00.6013501Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (0)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244151",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:48:58.7730429Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:48:58.7732028Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:48:58.7733612Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:48:58.7735372Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:48:58.7736435Z ===== 1969 passed, 3 skipped, 302 warnings, 10 errors in 869.17s (0:14:29) =====\n2026-11-22T10:49:00.9397462Z - JUnit failure count differs from phase reports\n2026-11-22T10:49:00.9414148Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:49:00.9590057Z ##[error]Process completed with exit code 1.\n2026-11-22T10:49:01.6759488Z - JUnit failure count differs from phase reports\n2026-11-22T10:49:01.6775040Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:49:01.6961542Z ##[error]Process completed with exit code 2.\n2026-11-22T10:49:03.2866150Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (1)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244152",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:40:49.9524347Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:40:49.9525592Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:40:49.9526829Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:40:49.9528066Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:40:49.9528883Z ==== 1455 passed, 41 skipped, 272 warnings, 10 errors in 392.83s (0:06:32) =====\n2026-11-22T10:40:51.5998396Z - JUnit failure count differs from phase reports\n2026-11-22T10:40:51.6011841Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:40:51.6163305Z ##[error]Process completed with exit code 1.\n2026-11-22T10:40:52.0536955Z - JUnit failure count differs from phase reports\n2026-11-22T10:40:52.0545467Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:40:52.0718040Z ##[error]Process completed with exit code 2.\n2026-11-22T10:40:53.6454179Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (5)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244155",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:43:41.4608570Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:43:41.4611649Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:43:41.4614820Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:43:41.4617746Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:43:41.4619594Z ========== 1757 passed, 230 warnings, 10 errors in 547.69s (0:09:07) ===========\n2026-11-22T10:43:43.5276354Z - JUnit failure count differs from phase reports\n2026-11-22T10:43:43.5303366Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:43:43.5466937Z ##[error]Process completed with exit code 1.\n2026-11-22T10:43:44.2478076Z - JUnit failure count differs from phase reports\n2026-11-22T10:43:44.2495144Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:43:44.2653696Z ##[error]Process completed with exit code 2.\n2026-11-22T10:43:46.0106622Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (6)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244156",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:40:02.4661553Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:40:02.4662703Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:40:02.4663855Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:40:02.4665075Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:40:02.4665916Z ===== 1680 passed, 6 skipped, 150 warnings, 10 errors in 340.21s (0:05:40) =====\n2026-11-22T10:40:03.7251476Z - JUnit failure count differs from phase reports\n2026-11-22T10:40:03.7269077Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:40:03.7398134Z ##[error]Process completed with exit code 1.\n2026-11-22T10:40:04.2530214Z - JUnit failure count differs from phase reports\n2026-11-22T10:40:04.2542138Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:40:04.2688601Z ##[error]Process completed with exit code 2.\n2026-11-22T10:40:05.9136177Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (2)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244168",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:42:40.9596715Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:42:40.9598150Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:42:40.9599493Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:42:40.9600887Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:42:40.9601810Z ===== 2020 passed, 9 skipped, 193 warnings, 10 errors in 491.20s (0:08:11) =====\n2026-11-22T10:42:42.9678027Z - JUnit failure count differs from phase reports\n2026-11-22T10:42:42.9698401Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:42:42.9851645Z ##[error]Process completed with exit code 1.\n2026-11-22T10:42:43.7442490Z - JUnit failure count differs from phase reports\n2026-11-22T10:42:43.7456383Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:42:43.7627701Z ##[error]Process completed with exit code 2.\n2026-11-22T10:42:45.2679380Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (7)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244186",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:48:54.9486182Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:48:54.9488201Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:48:54.9490213Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:48:54.9492468Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:48:54.9493710Z ==== 1527 passed, 34 skipped, 187 warnings, 10 errors in 851.51s (0:14:11) =====\n2026-11-22T10:48:57.2580329Z - JUnit failure count differs from phase reports\n2026-11-22T10:48:57.2603523Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:48:57.2767926Z ##[error]Process completed with exit code 1.\n2026-11-22T10:48:58.0189916Z - JUnit failure count differs from phase reports\n2026-11-22T10:48:58.0215237Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:48:58.0435055Z ##[error]Process completed with exit code 2.\n2026-11-22T10:48:59.6669137Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (3)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244199",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:59:44.7497847Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:59:44.7499448Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:59:44.7500665Z = 2 failed, 6632 passed, 10 skipped, 548 warnings, 10 errors in 1517.09s (0:25:17) =\n2026-11-22T10:59:47.6211889Z Selected: 6644; outcomes: `{\"failed\": 2, \"passed\": 6632, \"skipped\": 10}`.\n2026-11-22T10:59:47.6223240Z - JUnit failure count differs from phase reports\n2026-11-22T10:59:47.6239358Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:59:47.6407182Z ##[error]Process completed with exit code 1.\n2026-11-22T10:59:48.8725686Z Selected: 6644; outcomes: `{\"failed\": 2, \"passed\": 6632, \"skipped\": 10}`.\n2026-11-22T10:59:48.8737109Z - JUnit failure count differs from phase reports\n2026-11-22T10:59:48.8763545Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:59:48.8946684Z ##[error]Process completed with exit code 2.\n2026-11-22T10:59:50.3424348Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (9)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244218",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:45:29.2364166Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:45:29.2365784Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:45:29.2366841Z ===== 2180 passed, 3 skipped, 268 warnings, 10 errors in 641.70s (0:10:41) =====\n2026-11-22T10:45:31.4366782Z - JUnit failure count differs from phase reports\n2026-11-22T10:45:31.4379186Z | tests/e2e/tennis_scene/test_publication_failures.py | 2 | 0.004 | 18.245 | 0.003 |\n2026-11-22T10:45:31.4383680Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:45:31.4564413Z ##[error]Process completed with exit code 1.\n2026-11-22T10:45:32.2145925Z - JUnit failure count differs from phase reports\n2026-11-22T10:45:32.2158118Z | tests/e2e/tennis_scene/test_publication_failures.py | 2 | 0.004 | 18.245 | 0.003 |\n2026-11-22T10:45:32.2162556Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:45:32.2356903Z ##[error]Process completed with exit code 2.\n2026-11-22T10:45:33.7870254Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (8)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244227",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:45:07.3191750Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:45:07.3192951Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:45:07.3194135Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:45:07.3195349Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:45:07.3196174Z ========== 2494 passed, 194 warnings, 10 errors in 633.44s (0:10:33) ===========\n2026-11-22T10:45:09.6311376Z - JUnit failure count differs from phase reports\n2026-11-22T10:45:09.6323632Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:45:09.6470048Z ##[error]Process completed with exit code 1.\n2026-11-22T10:45:10.3328576Z - JUnit failure count differs from phase reports\n2026-11-22T10:45:10.3340327Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:45:10.3512112Z ##[error]Process completed with exit code 2.\n2026-11-22T10:45:14.7148855Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (4)",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137179244271",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T10:50:45.3760832Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:50:45.3762628Z ERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:50:45.3764410Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:50:45.3766186Z ERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.submodules.configuration.ViTPoseRuntimeConfig.engines declares a Python runtime default; defaults must be composition-owned.\n2026-11-22T10:50:45.3767477Z ===== 3278 passed, 1 skipped, 254 warnings, 10 errors in 972.39s (0:16:12) =====\n2026-11-22T10:50:47.5015023Z - JUnit failure count differs from phase reports\n2026-11-22T10:50:47.5033292Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:50:47.5182832Z ##[error]Process completed with exit code 1.\n2026-11-22T10:50:48.2468185Z - JUnit failure count differs from phase reports\n2026-11-22T10:50:48.2481897Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T10:50:48.2649744Z ##[error]Process completed with exit code 2.\n2026-11-22T10:50:50.1087345Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45012957049,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45012957049/job/137185649877",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-22T11:00:26.5269166Z     \"cpu-test-measurements-45012957049-shard-10: incomplete test observations: ['JUnit testcase count differs from execution/collection reports', 'JUnit failure count differs from phase reports', 'JUnit test identities differ from phase reports']\",\n2026-11-22T11:00:26.5271660Z     \"cpu-test-measurements-45012957049-shard-2: incomplete test observations: ['JUnit testcase count differs from execution/collection reports', 'JUnit failure count differs from phase reports', 'JUnit test identities differ from phase reports']\",\n2026-11-22T11:00:26.5274363Z     \"cpu-test-measurements-45012957049-shard-3: incomplete test observations: ['JUnit testcase count differs from execution/collection reports', 'JUnit failure count differs from phase reports', 'JUnit test identities differ from phase reports']\",\n2026-11-22T11:00:26.5277141Z     \"cpu-test-measurements-45012957049-shard-4: incomplete test observations: ['JUnit testcase count differs from execution/collection reports', 'JUnit failure count differs from phase reports', 'JUnit test identities differ from phase reports']\",\n2026-11-22T11:00:26.5279772Z     \"cpu-test-measurements-45012957049-shard-5: incomplete test observations: ['JUnit testcase count differs from execution/collection reports', 'JUnit failure count differs from phase reports', 'JUnit test identities differ from phase reports']\",\n2026-11-22T11:00:26.5282538Z     \"cpu-test-measurements-45012957049-shard-6: incomplete test observations: ['JUnit testcase count differs from execution/collection reports', 'JUnit failure count differs from phase reports', 'JUnit test identities differ from phase reports']\",\n2026-11-22T11:00:26.5284897Z     \"cpu-test-measurements-45012957049-shard-7: incomplete test observations: ['JUnit testcase count differs from execution/collection reports', 'JUnit failure count differs from phase reports', 'JUnit test identities differ from phase reports']\",\n2026-11-22T11:00:26.5287337Z     \"cpu-test-measurements-45012957049-shard-8: incomplete test observations: ['JUnit testcase count differs from execution/collection reports', 'JUnit failure count differs from phase reports', 'JUnit test identities differ from phase reports']\",\n2026-11-22T11:00:26.5289984Z     \"cpu-test-measurements-45012957049-shard-9: incomplete test observations: ['JUnit testcase count differs from execution/collection reports', 'JUnit failure count differs from phase reports', 'JUnit test identities differ from phase reports']\",\n2026-11-22T11:00:26.5292484Z     \"incomplete/failed test observations\",\n2026-11-22T11:00:26.5580473Z ##[error]Process completed with exit code 1.\n2026-11-22T11:00:26.5791032Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
