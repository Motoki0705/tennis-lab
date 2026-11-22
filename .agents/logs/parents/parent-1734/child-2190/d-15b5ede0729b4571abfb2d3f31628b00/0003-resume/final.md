[PR #2200](https://github.com/Motoki0705/tennis-lab/pull/2200)へpushしました。branchは`codex/court-acceptance-postprocess`、commitは`7b59c4730`です。今回、実装・閾値・凍結推定は変更せず、GPUも使用していません。

原画像15frame/cameraから白線中心の仮参照を作り、全276件を再評価しました。

| 条件 | 誤採用：旧参照→仮参照 | 正しい推定の採用率（仮） |
|---|---:|---:|
| 旧処理 | 144→105 | 66.2% |
| 採否のみ変更 | 175→116 | 84.6% |
| 後処理も変更 | 176→116 | 81.0% |

参照誤差による判定が多数ありました。一方、cam0・cam2には白線交点から約85px・61px離れた採用推定も残っています。仮評価でも目標未達です。

**新参照は未再承認です。** 全cameraの奥ベースライン未測定、cam3の1frame失敗、cam1/3の約5pxの時間変化を記録しました。[原解像度の確認資料](https://github.com/Motoki0705/tennis-lab/blob/7b59c4730dfa934fb8e2323ae0553b96f737f412/outputs/tennis_scene/evaluate/court-reference-white-lines-20261116/review/index.html)での再承認と、時間変化・未測定線の扱いを親へ判断事項として返します。

詳細は[knowledge 71](https://github.com/Motoki0705/tennis-lab/blob/7b59c4730dfa934fb8e2323ae0553b96f737f412/knowledge/nodes/court_detection/000071-run-court-white-line-reference-20261116.md)。Court valの114件喪失など、参照に依存しない回帰も未解決です。

数値テスト3件、knowledge検証、pre-commitは成功。[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43865648771)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43865648746)も対象HEADで成功しました。validatorは0回。PRはready for review、merge・issue closeはしていません。