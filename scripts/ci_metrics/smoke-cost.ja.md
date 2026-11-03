# モデルsmokeのコストと採否

[English](smoke-cost.md) · [简体中文](smoke-cost.zh-CN.md)

## 判断

[#1404](https://github.com/Motoki0705/tennis-lab/issues/1404) のSLCS学習・評価と
MDD構造テストは維持します。計測したモデル処理の寄与が小さく、接続の検証範囲を削ることや
meta device用の準備を追加する根拠はありません。merge fixture準備とCPUの分配を優先します。

## 計測（2026-10-25）

再現コマンドは[英語版](smoke-cost.md#observation-2026-10-25)を参照してください。
ローカルLinuxで2 worker、native thread各1、CUDA不可視として12件すべて成功しました。
1回の部分観測であり、全CPU baselineやGitHub-hosted runnerの性能ではありません。
source・環境、inventory、phase event、JUnitと完全性の照合結果を一緒に保存しました。
前回取り消された計測は再利用していません。

| 対象 | 件数 | call合計秒 |
|---|---:|---:|
| SLCS学習smoke（通常fit/callback含む） | 5 | 4.582 |
| MDD CNNテスト | 7 | 1.054 |
| 本番MDD config構造検査（上の内数） | 1 | 0.099 |

コマンド全体は単調時計で24.853秒、workerのcall合計は5.636秒でした。
全workerの収集完了まで14.347秒、最後のphase報告から終了まで7.248秒です。
workerの時間は重なり得るため、実時間に足しません。pytest単一processの最大RSSは
958,972 KiBで、合算メモリでも個別モデルのメモリでもありません。

## 維持する契約と再検討条件

テストの削除・移動・統合はありません。frames 6/18×context off/on、実adapterから
model・loss/backwardまでの接続、context勾配、optimizer更新、checkpoint再読込、
4評価modeのtarget/mask/frame/ID保存を維持します。通常fit、監視metricとcheckpoint
callback、実TensorBoard eventも維持します。MDDでは本番module treeを実構築し、
convolution数とSwiGLU幅を確認する試験をそのまま残します。

削除ケースがないため、その対応表や削除前後の変異比較は不要です。将来の全件計測で
寄与が大きくなった場合、本番構成の拡大で初期化コストが変わった場合に再検討します。
一般の計測・比較契約は[README](README.ja.md)を正本とします。
