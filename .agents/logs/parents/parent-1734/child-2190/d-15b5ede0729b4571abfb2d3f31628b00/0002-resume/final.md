[PR #2200](https://github.com/Motoki0705/tennis-lab/pull/2200)をReady for reviewにしました。**受入未達で、現案のmergeは推奨しません。**

境界・重複点の除外、可視線・外挿安定性による採否と順位、診断・cache契約を実装しました。承認参照と許容差を固定した全276件の結果です。

| 条件 | 誤採用 | 正しい表示候補の採用率 |
|---|---:|---:|
| 旧処理 | 144 | 8/12＝66.7% |
| 採否のみ変更 | 175 | 7/9＝77.8% |
| 後処理も変更 | 176 | 4/6＝66.7% |

新処理では正解候補がある40画像中4画像だけを救済しました。cam3の採用24件はすべて許容差外です。回帰も、幾何成立がCourt valで **2,211→2,097件**、synthetic testで **573→558件**に減りました。

- branch: `codex/court-acceptance-postprocess`
- HEAD: `b533eb7ab`
- 新規テスト17ケース。関連テスト・pre-commit・[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43846256607)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43846256489)成功。validatorは0回。
- [詳細・camera別集計・画像](https://github.com/Motoki0705/tennis-lab/blob/b533eb7ab/knowledge/nodes/court_detection/000070-run-court-acceptance-implementation-20261116.md)を保存済み。

実画像testは未定義、synthetic testと学習集合の非重複は未確認です。既存sceneのGTも限定的です。

原画像の塗線による独立した位置合わせと、支持点選択の再設計を提案します。現scoreを#2191の精度保証には使えません。merge・issue closeは行っていません。