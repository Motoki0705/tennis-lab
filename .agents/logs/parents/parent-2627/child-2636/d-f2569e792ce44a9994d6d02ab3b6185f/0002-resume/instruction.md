## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T112440Z_a93481dc77bb_meiji-stopped-clip-evidence",
      "name": "meiji-stopped-clip-evidence",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 72.778,
      "log": "[local path]",
      "log_excerpt": "clip_002: 12 synchronized frames, player_association_ambiguous_players\nclip_018: 12 synchronized frames, player_association_ambiguous_players\nclip_020: 9 synchronized frames, player_association_player_not_found\nclip_023: 12 synchronized frames, player_association_ambiguous_players\nclip_028: 11 synchronized frames, player_association_ambiguous_players\nclip_038: 12 synchronized frames, player_association_ambiguous_players\nclip_036: 12 synchronized frames, player_association_ambiguous_players\nclip_039: 9 synchronized frames, player_association_ambiguous_players\nclip_049: 12 synchronized frames, player_association_ambiguous_association\nclip_058: 12 synchronized frames, player_association_player_not_found\nclip_060: 9 synchronized frames, player_association_ambiguous_association\nCompleted 14 clips, 154 synchronized frames, 616 camera frames in 70.09s"
    }
  ]
}
