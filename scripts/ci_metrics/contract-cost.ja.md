# 入力契約テストの整理

[English](contract-cost.md) · [简体中文](contract-cost.zh-CN.md)

[#1403](https://github.com/Motoki0705/tennis-lab/issues/1403) では、異常分類は共通validatorで
検証し、各public入口・実モデル設定が計算前にそのvalidatorを通ることを別に確認します。
本番コード、数値許容誤差、既存のstate・排他条件の検査は変えていません。

| 旧ケース | 維持先と相互作用 | 採否 |
|---|---|---|
| dense mask異常5分類×MHA/GQA×module/direct/update＝30 | 共通validatorの5分類と6入口/方式の実検証・計算未到達。dtype、厳密rank/shape、broadcast拒否、deviceを維持 | 11件、block構築30→6 |
| direct/moduleの正常出力一致3方式 | 既存の値・入力/parameter勾配・state一致検査がdense側を包含。CSWAのall-valid分岐は独立に維持 | 弱いdense側2件を統合 |
| RGB異常6分類×DPT/DeepQuery×channels 2/5＝24 | 共通validatorで旧6分類とrank/width/最小正常入力。全4実型/channel入口で実際のvalidator接続とforward未到達 | 13件、detector構築24→4 |

完全なfile/function対応は[英語版の表](contract-cost.md)を参照してください。
対象全体は旧114件・新82件とも成功し、差引32件の削減です。単なる層の移動ではありません。
候補内のblock/detector構築は57→11となり、mutableモデルの共有はありません。

## 不具合検出と計測

| 一時的に注入した不具合 | 旧版の失敗数 | 新版の失敗数 |
|---|---:|---:|
| dense maskのfloat許容 | 6 | 7 |
| decoded RGBのfloat許容 | 4 | 5 |
| DPT入口のvalidator迂回 | 12 | 2 |

全6実行が終了値1の通常テスト失敗となり、収集・setupエラーはありません。
spyは入口に実際に束縛された関数を包みます。既知の正常validatorへ置き換えると接続の
欠陥を隠してしまうためです。DPT迂回は両channel設定で検出しています。

版・件数・phase時間・変異結果は[機械可読の証拠](contract-evidence.json)に記録しています。
計測は各版1回、xdist workerなし、native thread各1、CUDA不可視のローカル部分観測です。
別の検証も実行中で、起動/cache/負荷は統制していません。CI実時間の改善率や親の
warm-cache中央値・p95目標の達成とは扱いません。再現手順は[英語版](contract-cost.md)、
観測の一般契約は[README](README.ja.md)を参照してください。
