## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T115210Z_4a7282d419c9_meiji-stopped-clip-replay-cpu",
      "name": "meiji-stopped-clip-replay-cpu",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 114.494,
      "log": "[local path]",
      "log_excerpt": "clip_020: unchanged=player_association_player_not_found\nclip_004: unchanged=player_association_ambiguous_players, delete_reviewed_dino_rows=player_association_ambiguous_players, omit_runner_up_group=court_side_no_consistent_hypothesis\nclip_036: unchanged=player_association_ambiguous_players, delete_reviewed_dino_rows=player_association_ambiguous_players, join_reviewed_groups=player_association_ambiguous_players, omit_runner_up_group=passed\nclip_038: unchanged=player_association_ambiguous_players, delete_reviewed_dino_rows=player_association_ambiguous_players, join_reviewed_groups=player_association_ambiguous_players, omit_runner_up_group=passed\nclip_039: unchanged=player_association_ambiguous_players, omit_runner_up_group=passed\nclip_002: unchanged=player_association_ambiguous_players, delete_reviewed_dino_rows=player_association_ambiguous_players, omit_runner_up_group=passed\nclip_023: unchanged=player_association_ambiguous_players, delete_reviewed_dino_rows=player_association_ambiguous_players, omit_runner_up_group=passed\nclip_060: unchanged=player_association_ambiguous_association\nclip_049: unchanged=player_association_ambiguous_association, delete_reviewed_dino_rows=player_association_ambiguous_association\nclip_028: unchanged=player_association_ambiguous_players, delete_reviewed_dino_rows=player_association_ambiguous_players, omit_runner_up_group=passed\nclip_058: unchanged=player_association_player_not_found, delete_reviewed_dino_rows=player_association_player_not_found\nReproduced 14/14 stop codes on CPU in 102.60s"
    }
  ]
}
