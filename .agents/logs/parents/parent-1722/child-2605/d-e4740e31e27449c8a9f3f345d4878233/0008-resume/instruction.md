## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T065715Z_94131959dda0_scene-events-v3-rebuild-555a246",
      "name": "scene-events-v3-rebuild-555a246",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 2362.488,
      "log": "[local path]",
      "log_excerpt": "      \"net_events\": 5\n    }\n  }\n}\ntrue\ndata/tennis_multiview/processed/meiji_4cam/benchmarks/.multiview-v3-555a246aa/manifest.json: OK\ndata/tennis_multiview/processed/meiji_4cam/benchmarks/.multiview-v3-555a246aa/SHA256SUMS: OK\ndata/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v2/manifest.json: OK\ndata/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v2/SHA256SUMS: OK\ndata/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3/manifest.json: OK\ndata/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3/SHA256SUMS: OK\nVerified release candidate matches the previous annotation counts and complete split assignments"
    }
  ]
}
