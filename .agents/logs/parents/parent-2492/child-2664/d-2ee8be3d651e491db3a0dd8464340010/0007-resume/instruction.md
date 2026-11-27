## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261127T003300Z_3ec0e6d1773a_i2664-full-config-final",
      "name": "i2664-full-config-final",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 382.379,
      "log": "[local path]",
      "log_excerpt": "1.40s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_side_rejects_missing_video_wrong_digest_player_evidence_and_changed_camera_or_media\n1.40s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_alternative_people_artifacts_preserve_the_3d_input_and_publication_contract[rfdetr-l-rtmpose-x]\n1.39s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_cli_writes_candidates_from_only_raw_ports_and_pinned_side\n1.39s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_empty_observations_require_enabled_person_inputs[dino-disabled]\n1.39s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_legacy_dependencies_are_rejected_at_each_input_level[person_tracking/cam0-person_tracks-person_pose_features/cam0]\n1.38s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_empty_observations_require_enabled_person_inputs[enabled-empty]\n1.38s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_legacy_dependencies_are_rejected_at_each_input_level[player_association-person_identities-person_detection/cam0]\n1.37s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_empty_observations_require_enabled_person_inputs[vitpose-disabled]\n1.37s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_alternative_people_artifacts_preserve_the_3d_input_and_publication_contract[dino-rtmpose-x]\n1.37s setup    tests/integration/tennis_scene/test_multiview_pose_store.py::test_alternative_people_artifacts_preserve_the_3d_input_and_publication_contract[rfdetr-l-vitpose-h]\n======================= 641 passed in 282.11s (0:04:42) ========================\nStrict configuration/path audit passed against current source; 140 runtime boundaries verified."
    }
  ]
}
