## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T044546Z_0c1fb6689a8e_scene-events-v3",
      "name": "scene-events-v3",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 357.819,
      "log": "[local path]",
      "log_excerpt": "10.02s call     tests/e2e/tennis_scene/test_publication_failures.py::test_publication_failure_exits_nonzero_and_keeps_failure_record[single]\n9.78s call     tests/e2e/tennis_scene/test_publication_failures.py::test_publication_failure_exits_nonzero_and_keeps_failure_record[dataset]\n8.54s call     tests/e2e/tennis_scene/test_video_batch_cli.py::test_cli_runs_one_video_and_exits_nonzero_for_unresolved_failure[True]\n8.44s call     tests/e2e/tennis_scene/test_video_batch_cli.py::test_cli_runs_one_video_and_exits_nonzero_for_unresolved_failure[False]\n8.37s call     tests/e2e/tennis_scene/test_video_court_cli.py::test_single_clip_cli_loads_pinned_shared_geometry_or_records_failure[True]\nFAILED tests/integration/tennis_scene/test_fasthmr_recipe.py::test_only_the_exact_standard_recipe_patch_preserves_previous_source_identity\nFAILED tests/e2e/tennis_scene/test_video_batch_cli.py::test_cli_runs_one_video_and_exits_nonzero_for_unresolved_failure[False]\nFAILED tests/e2e/tennis_scene/test_video_batch_cli.py::test_cli_runs_one_video_and_exits_nonzero_for_unresolved_failure[True]\nFAILED tests/integration/tasks/annotation/test_scene_workflow.py::test_blank_cli_pins_policy_runs_isolated_condition_and_reports_without_publication\nFAILED tests/e2e/tasks/annotation/test_workflow.py::test_scene_entrypoint_annotates_continues_reloads_triangulates_and_publishes\nFAILED tests/unit/tasks/slcs/data/test_annotation.py::test_fixture_publishes_through_the_component_store\n================== 6 failed, 1806 passed in 354.70s (0:05:54) =================="
    }
  ]
}
