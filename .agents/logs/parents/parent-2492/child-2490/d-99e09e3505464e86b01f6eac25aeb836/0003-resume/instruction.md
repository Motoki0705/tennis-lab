## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T122655Z_02cca93c039e_sapiens2-runtime",
      "name": "sapiens2-runtime",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 139.732,
      "log": "[local path]",
      "log_excerpt": "  × Failed to download `soupsieve==4.1.3`\n  ├─▶ Request failed after 3 retries in 124.9s\n  ├─▶ Failed to fetch:\n  ├─▶ error sending request for url"
    },
    {
      "job_id": "20261124T122655Z_ee8741870b59_sapiens2-1b-weight",
      "name": "sapiens2-1b-weight",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 1854.043,
      "log": "[local path]",
      "log_excerpt": "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n./outputs/sapiens2/weights/sapiens2_1b_pose.safetensors\nsapiens2_1b_pose.safetensors: OK\nsapiens2_1b_pose.safetensors 6079194752 bytes"
    },
    {
      "job_id": "20261124T122655Z_c5756afdd2ff_sapiens2-5b-weight",
      "name": "sapiens2-5b-weight",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 2846.901,
      "log": "[local path]",
      "log_excerpt": "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n./outputs/sapiens2/weights/sapiens2_5b_pose.safetensors\nsapiens2_5b_pose.safetensors: OK\nsapiens2_5b_pose.safetensors 20480899148 bytes"
    }
  ]
}

## 親からの補足: 土台の更新（採点器の修正）

待機明けの自動再開は、#2495 のPR #2498 の最新head `0d4bf0990`（正式レビューの指摘への修正）がこのbranchに含まれていないため、親CLIに拒否された。親がこのbranchへ取り込んだ（このターンの開始時点のHEAD）。#2496 のPR #2497 は d64962a28 のまま。

修正の内容: 同じIoUの候補が複数あるとき、候補の行順だけで姿勢スコア（OKS・PCK）が変わる問題を直した。採点前に、候補を「検出スコアの高い順 → BBox座標の辞書順 → 関節配列の辞書順」に並べてから照合する。詳しくは `src/tennis_scene/evaluation/player_pose/README.md` を参照する。

- このbranchで、すでに修正前の採点器で出した精度の値があれば、保存済みのraw予測から新しい採点器で再採点し（GPU不要）、記録を更新する。値が変わったかどうかも書く。
- 取り込み後のHEADで影響範囲のテストを通してから進める。PRがあるならpushする。
- 待機（`PARENT_RESULT` の waiting）で終える前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してから終える。最終行の `PARENT_RESULT` は1行の正しいJSONにする。
