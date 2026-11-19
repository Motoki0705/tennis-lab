最新main `dcd653459` を `codex/person-stage-speed` に取り込み、PR #2240の衝突を解消しました。HEADは `3f47e1bc7176318477f58279aade1db1d1deba39` です。 [PR #2240](https://github.com/Motoki0705/tennis-lab/pull/2240)

`definition.py` と `orchestrator.py` では、#2244の共有court sideの検証と、人物モデルの精度設定・公開再開時の精度照合を組み合わせました。DINO拡張のpinと再現手順も引き継いでいます。

既定FP32のソースfingerprintは最新mainと一致し、merge前との差分はmain由来の22ファイルだけでした。最新mainが保存した31段の成果物は、既定・明示FP32の両方で再利用できました。

`tests/unit/tennis_scene` 全1,061件を含むCPUテスト1,152件とpre-commitが通過しました。対象HEADのPR CI・手動CIはともに `Python` を含む全14ジョブが成功し、Knowledge libraryも通過しました。結果はPR本文に追記済みです。 [PR CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44313549171)・[手動CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44313644698)・[Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44313549110)