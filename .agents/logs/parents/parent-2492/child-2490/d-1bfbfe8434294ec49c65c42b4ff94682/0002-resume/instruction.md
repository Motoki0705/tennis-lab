## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204479371993481292_2026-11-24T11-43-48-006518707Z_1190023_i2490-rtmpose-m-pt32-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204479371629885537_2026-11-24T11-43-48-370114462Z_1190093_i2490-rtmpose-m-pt16-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204479371298644878_2026-11-24T11-43-48-701355121Z_1190188_i2490-rtmpose-m-trt16-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204479370898726033_2026-11-24T11-43-49-101273966Z_1190269_i2490-rtmpose-x-pt32-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204479370537743155_2026-11-24T11-43-49-462256844Z_1190346_i2490-rtmpose-x-pt16-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204479370187195038_2026-11-24T11-43-49-812804961Z_1190424_i2490-rtmpose-x-trt16-accuracy.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}

## 親からの補足: 土台の更新

待機明けの自動再開は、#2495 のPR #2498 と #2496 のPR #2497 の最新headがこのbranchに含まれていないため、親CLIに拒否された。親がこのbranchへ両方の最新head（#2498: b1011e406、#2497: d64962a28）をmergeした（このターンの開始時点のHEAD）。#2497 の d64962a28 は scene cache 互換を直す変更（`tests/integration/tennis_scene/test_fp32_cache_compatibility.py` 関連）を含む。

- 取り込み後のHEADで影響範囲のテストを通してから先へ進むこと。PRがあるならpushする。
- 土台はまだdraftで、今後も更新されうる。待機（`PARENT_RESULT` の waiting）で終える前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してから終えること。
- 最終行の `PARENT_RESULT` は1行の正しいJSONにする（括弧の数を確認する）。
