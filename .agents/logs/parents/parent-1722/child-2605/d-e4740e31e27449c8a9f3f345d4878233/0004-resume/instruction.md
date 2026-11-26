## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T053752Z_66d354fe0c1e_scene-events-v3-annotation-pidfd",
      "name": "scene-events-v3-annotation-pidfd",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 83.704,
      "log": "[local path]",
      "log_excerpt": "configfile: pyproject.toml\nplugins: cov-7.1.0, hydra-core-1.3.6, anyio-4.13.0, xdist-3.8.0\ncollected 2 items\n\ntests/integration/tasks/annotation/test_scene_workflow.py .              [ 50%]\ntests/e2e/tasks/annotation/test_workflow.py .                            [100%]\n\n- generated xml file: ./outputs/scene-events-v3/pytest-annotation-python31117.xml -\n============================= slowest 2 durations ==============================\n46.42s call     tests/e2e/tasks/annotation/test_workflow.py::test_scene_entrypoint_annotates_continues_reloads_triangulates_and_publishes\n17.03s call     tests/integration/tasks/annotation/test_scene_workflow.py::test_blank_cli_pins_policy_runs_isolated_condition_and_reports_without_publication\n========================= 2 passed in 80.08s (0:01:20) ========================="
    }
  ]
}
