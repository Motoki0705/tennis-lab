# 再開（親 #2492 から）: Sapiens2 License への同意

ユーザーが、評価目的で Sapiens2 License（固定版 https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/LICENSE.md ）に同意した（2026-11-24、親 #2492 本文の「ユーザー判断」に記録済み）。ONNX/TensorRTへの形式変換は、改変・派生物の作成として許可範囲に含まれる。

## 進め方

- 公式重みでの推論、TensorRT変換、1B/5B × PT32/PT16/TRT16 の精度・速度・VRAMの計測に進む。GPUはtraining queue経由で使う。
- 5BのFP32は、合意どおりローカルGPUで実際にCUDAロードを試し、OOMを実行不能として必要メモリの見積もりとともに記録する。FP16やCPU offloadの結果をFP32の実測として扱わない。
- 速度は #2496 の固定入力とharnessで測る。#2497 のhead d64962a28 で固定入力が使えなければ、何が欠けているかを最終報告に書いて親へ返す。
- ライセンス記録（knowledgeのsapiens2.md）に次を書く。
  - ユーザーが評価目的で同意したこと
  - 本番採用前の確認事項3点: 禁止用途「biometric processing」条項が本件の姿勢推定に当たるかの解釈、再配布時のライセンス同梱、論文発表時のSapiens2使用の明記
- 取得した重みとengineはrepoにcommitしない。

## 土台の更新

このbranchには、すでに #2498 head b1011e406 と #2497 head d64962a28 が入っている。土台は今後も更新される。待機（`PARENT_RESULT` の waiting）で終える前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してから終えること。両PRの最新headが含まれていないと、待機明けの自動再開は拒否される。
