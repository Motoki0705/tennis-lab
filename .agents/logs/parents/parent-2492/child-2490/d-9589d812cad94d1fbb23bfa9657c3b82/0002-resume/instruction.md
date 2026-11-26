## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T113813Z_16d5e8631dce_i2490-pmpose-weights",
      "name": "i2490-pmpose-weights",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 565.08,
      "log": "[local path]",
      "log_excerpt": "49a070a6f8396147323b1c4959474077dbe2ce8e  outputs/pmpose/BBoxMaskPose\n931b2f14486c199feee57714d6ab7e162fb904495050efa4d2142306395dc250  outputs/pmpose/PMPose-l-1.0.0.pth\nca3e5577839e5ce22390d1d6409241f61526346c80086d3ee1990b503dd93635  outputs/pmpose/PMPose-h-1.0.0.pth"
    },
    {
      "job_id": "20261124T113813Z_29ea3f805bfa_i2490-pmpose-deps",
      "name": "i2490-pmpose-deps",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 328.703,
      "log": "[local path]",
      "log_excerpt": "Failed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'. Please make sure you already `pip install kernels` and the kernels >= 0.11.1\nTraceback (most recent call last):\nTypeError: NoneType takes no arguments"
    }
  ]
}

## 親からの補足: 土台の更新

待機明けの自動再開は、#2495 のPR #2498 と #2496 のPR #2497 の最新headがこのbranchに含まれていないため、親CLIに拒否された。親がこのbranchへ両方の最新head（#2498: b1011e406、#2497: d64962a28）をmergeした（このターンの開始時点のHEAD）。#2497 の d64962a28 は scene cache 互換を直す変更（`tests/integration/tennis_scene/test_fp32_cache_compatibility.py` 関連）を含む。

- 取り込み後のHEADで影響範囲のテストを通してから先へ進むこと。PRがあるならpushする。
- 土台はまだdraftで、今後も更新されうる。待機（`PARENT_RESULT` の waiting）で終える前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してから終えること。
- 最終行の `PARENT_RESULT` は1行の正しいJSONにする（括弧の数を確認する）。
