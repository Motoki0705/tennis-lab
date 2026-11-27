RF-DETR-L の検出閾値は **0.25**、RTMPose-X の関節 confidence は **生の SimCC スコア**を使う案です。検出閾値と関節スコアの扱いを新モデルの設定・成果識別に記録し、この条件で #2487 の下流比較に進めてよいか判断をお願いします。

保存済みの TRT16 予測を SHA-256 照合後に再採点しました。RF-DETR-L の閾値を 0.25 から 0.30 に上げると、6,844 画像で 2,917 box が外れ、選手の疑似 GT に対する Recall@IoU 0.50 は **96.81% → 96.66%** でした。同じ 26,276 box で、RTMPose-X と ViTPose-H の関節スコアの中央値は **0.691／0.795** で、どちらも 1 を超える値がありました。 [再集計と棚卸しの記録](https://github.com/Motoki0705/tennis-lab/issues/2664#issuecomment-7195383283)

| 下流 | スコアの用途と確認結果 |
|---|---|
| 検出・重複 box 統合 | DINO の閾値は 0.30。任意の統合は IoU ≥ 0.8 で score 順に採用。#1739 は元の検出行をすべて要求する。 |
| StrongSORT++・CLIP | 検出 score を Kalman 更新の `1 − score` に使用。pose 距離は両側で 0.30 以上の共通4関節を要求し、confidence で重み付けする。CLIP は box の画像を使う。 |
| 選手選別・カメラ間対応 | 足元位置、コート内滞在、外観を使う。検出行と追跡結果を介してスコアの影響を受ける。 |
| 観測 mask・校正・標準3D復元 | 実観測かつ画像内で 0.15 以上を可視とし、校正と三角測量では 0.30 以上を使用。三角測量と身体配置は confidence を重みに使う。 |
| #1739 の3D候補・追跡 | 候補レイは 0.25 以上、強い観測は 0.55 以上。後者の通過率は RTMPose-X **80.44%**、ViTPose-H **88.02%**。 |
| 代表 view・GVHMR | 観測の被覆率と平均 confidence で view を選ぶ。GVHMR 内部の `> 0.50` を通る関節は RTMPose-X **86.53%**、ViTPose-H **91.21%**。 |
| SceneResult・可視化・学習 | `human_kp_vis` にマスク済み confidence を保存する。可視化は主に 0.15、SLCS は視点・関節平均 0.30 を既定の閾値に使う。 |

**統合順も判断をお願いします。** `multiview_pose` の DINO 固定条件は現行 main 上で拡張できます。#1739 の3D追跡から GVHMR・公開までを通す受入確認には、現在 open の #2470 が必要です。main を基点に実装を続け、#2470 のマージ後に新経路を1クリップで照合する案です。先に通し確認が必要なら #2470 の上に積む選択肢があります。 [PR #2470](https://github.com/Motoki0705/tennis-lab/pull/2470)

独立して進められる重み取得 CLI と3言語のライセンス記載を、`codex/parent-2492-issue-2664-primary` の `24d7e023ee7fb45a777111917defb457cef93cfa` にコミットしました。取得済みの公式重み2件の SHA-256、CLI の配置・照合動作、pytest 53件（新規4件）、pre-commit の ruff・mypy を確認済みです。全体の設定インベントリテストは収集に時間がかかり中断し、新規 CLI の登録を個別に確認しました。モデル統合、公式 MMPose・GPU 照合、1クリップ実行は残っており、担当 PR は未作成です。

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}