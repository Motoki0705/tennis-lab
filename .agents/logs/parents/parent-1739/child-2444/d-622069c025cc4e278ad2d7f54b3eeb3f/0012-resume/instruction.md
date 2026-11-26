## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T072946Z_ab757898a5ed_i2444-merge-cpu",
      "name": "i2444-merge-cpu",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 393.704,
      "log": "[local path]",
      "log_excerpt": "=================================== FAILURES ===================================\nE       AssertionError: assert {'submodules/...e3c82d2', ...} == {'submodules/...e3c82d2', ...}\ntests/integration/tennis_scene/test_fasthmr_recipe.py:162: AssertionError\n8.45s call     tests/unit/tennis_scene/generate_dataset/test_entrypoint.py::test_real_hydra_cli_fails_when_a_clip_fails\n6.42s call     tests/integration/tennis_scene/test_video_batch.py::test_interruption_and_device_failure_stop_without_partial_markers_then_resume[failure2-False]\nFAILED tests/integration/tennis_scene/test_fasthmr_recipe.py::test_only_the_exact_standard_recipe_patch_preserves_previous_source_identity\n================== 1 failed, 1668 passed in 391.08s (0:06:31) =================="
    }
  ]
}
