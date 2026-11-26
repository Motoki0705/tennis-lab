## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T045622Z_42ace67bac5c_scene-events-v3-regression",
      "name": "scene-events-v3-regression",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 375.373,
      "log": "[local path]",
      "log_excerpt": "E         json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\nE        +  where 1 = CompletedProcess(args=['./.venv/bin/...\"Expecting value\", s, err.value) from None\\njson.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)[local path]').returncode\ntests/e2e/tasks/annotation/test_workflow.py:132: AssertionError\n11.62s call     tests/unit/tennis_scene/generate_dataset/test_entrypoint.py::test_real_hydra_cli_fails_when_a_clip_fails\n10.66s call     tests/e2e/tennis_scene/test_video_court_cli.py::test_single_clip_cli_loads_pinned_shared_geometry_or_records_failure[False]\n10.36s call     tests/e2e/tennis_scene/test_publication_failures.py::test_publication_failure_exits_nonzero_and_keeps_failure_record[single]\n10.35s call     tests/e2e/tennis_scene/test_publication_failures.py::test_publication_failure_exits_nonzero_and_keeps_failure_record[dataset]\n10.07s call     tests/e2e/tennis_scene/test_video_batch_cli.py::test_cli_runs_one_video_and_exits_nonzero_for_unresolved_failure[False]\n9.67s call     tests/e2e/tennis_scene/test_video_batch_cli.py::test_cli_runs_one_video_and_exits_nonzero_for_unresolved_failure[True]\nFAILED tests/integration/tasks/annotation/test_scene_workflow.py::test_blank_cli_pins_policy_runs_isolated_condition_and_reports_without_publication\nFAILED tests/e2e/tasks/annotation/test_workflow.py::test_scene_entrypoint_annotates_continues_reloads_triangulates_and_publishes\n================== 2 failed, 1811 passed in 372.03s (0:06:12) =================="
    },
    {
      "job_id": "20261126T045622Z_0ea436a75cb5_scene-events-v3-benchmark",
      "name": "scene-events-v3-benchmark",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 2242.918,
      "log": "[local path]",
      "log_excerpt": "      \"net_clips\": 5,\n      \"net_events\": 6\n    },\n    \"test\": {\n      \"clips\": 11,\n      \"net_clips\": 5,\n      \"net_events\": 5\n    }\n  }\n}\ndata/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3/manifest.json: OK\ndata/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3/SHA256SUMS: OK"
    }
  ]
}
