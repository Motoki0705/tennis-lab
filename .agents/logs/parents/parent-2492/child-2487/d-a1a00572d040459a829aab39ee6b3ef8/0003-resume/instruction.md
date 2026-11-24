## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204478614370665283_2026-11-24T11-56-25-629334716Z_1278513_i2487-ecdet-l-coco-pt32-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204478614094424114_2026-11-24T11-56-25-905575885Z_1278586_i2487-ecdet-l-coco-pt16-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204478613818101458_2026-11-24T11-56-26-181898541Z_1278653_i2487-ecdet-l-coco-trt16-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204478613537376598_2026-11-24T11-56-26-462623401Z_1278721_i2487-ecpose-l-o365-trt16-conversion.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204478613232582331_2026-11-24T11-56-26-767417668Z_1278785_i2487-ecdet-x-coco-pt32-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204478612964169527_2026-11-24T11-56-27-035830472Z_1278865_i2487-ecdet-x-coco-pt16-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204478612683101748_2026-11-24T11-56-27-316898251Z_1278927_i2487-ecdet-l-o365-pt32-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204478612392360056_2026-11-24T11-56-27-607639943Z_1278996_i2487-ecdet-l-o365-pt16-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204478612081747949_2026-11-24T11-56-27-918252050Z_1279066_i2487-ecdet-x-o365-pt32-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204478611808412052_2026-11-24T11-56-28-191587947Z_1279137_i2487-ecdet-x-o365-pt16-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}

## 親からの補足: 土台の更新（採点器の修正）

待機明けの自動再開は、#2495 のPR #2498 の最新head `0d4bf0990`（正式レビューの指摘への修正）がこのbranchに含まれていないため、親CLIに拒否された。親がこのbranchへ取り込んだ（このターンの開始時点のHEAD）。#2496 のPR #2497 は d64962a28 のまま。

修正の内容: 同じIoUの候補が複数あるとき、候補の行順だけで姿勢スコア（OKS・PCK）が変わる問題を直した。採点前に、候補を「検出スコアの高い順 → BBox座標の辞書順 → 関節配列の辞書順」に並べてから照合する。詳しくは `src/tennis_scene/evaluation/player_pose/README.md` を参照する。

- このbranchで、すでに修正前の採点器で出した精度の値があれば、保存済みのraw予測から新しい採点器で再採点し（GPU不要）、記録を更新する。値が変わったかどうかも書く。
- 取り込み後のHEADで影響範囲のテストを通してから進める。PRがあるならpushする。
- 待機（`PARENT_RESULT` の waiting）で終える前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してから終える。最終行の `PARENT_RESULT` は1行の正しいJSONにする。
