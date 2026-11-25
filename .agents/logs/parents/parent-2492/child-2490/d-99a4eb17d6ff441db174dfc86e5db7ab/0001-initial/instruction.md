# #2490 統合担当（primary）: Top-down 2D poseの比較の統合（親 #2492）

担当issue #2490 の本文が仕様の正本。下の共通事項（親 #2492）が本文の「全69 clip」などより優先する。

## 担当の分担（同じissueに複数担当）

- primary（あなた）: 現行ViTPose-H（vitpose-h-multi-coco.pth、flip testあり・なし）を #2495 の疑似BBoxで精度採点し、PT32/PT16/TRT16の速度は #2496 の基準値を使う。全候補の比較表・関節別・難条件別の集計・Pareto frontier・ViTPose-Hに対する改善率・COCO17契約（現行 `Pose2DResult` の (F,17,3)）への適合確認・代表clipの同期横並び可視化・knowledge記録・人間向け報告のまとめ。補助担当の成果（commit・測定結果）を自分のPRへ統合し、このissueを閉じるPRを1本にする。
- 補助担当 `rtmpose`: RTMPose-M / RTMPose-X。
- 補助担当 `pmpose`: PMPose-L / PMPose-H（BBoxMaskPose v2 のBBox指定pose単体を主比較）。
- 補助担当 `sapiens2`: Sapiens2-1B / Sapiens2-5B（308関節からCOCO17を取り出す）。
- 候補ごとのadapterの置き場所と共通の入出力の形は、最初にあなたが決めて #2490 にコメントし、補助担当がそれに従えるようにする（補助担当も同時に動き始めるので早めに書く）。#2487 も同じ疑似GT・採点器を使うので、同じ採点の口を使う。

## 着手順

1. adapterの置き場所・入出力の形を決めてissueにコメントする。
2. 現行ViTPose-Hの精度採点を行う。
3. 補助担当の結果を順に統合し、比較表・可視化を更新する。

PRは早めにdraftで作る。queueへ計測jobを `add` したらjob IDを宣言して終了してよい。

## 親 #2492 の共通事項（3子共通）

- 代表clip（2026-11-24 ユーザー合意）: video_000 の clip_003・clip_057・clip_004・clip_013・clip_040・clip_028。評価対象は各clipでGVHMRが実際に推論したframe（約30fps格子）×cam0〜cam3。起票時の「全69 clip」は母集団と読み替える。clip一覧・疑似GT・入力manifestは #2495 が作るものを、同じID・checksumで参照する（自分で作り直さない）。
- 速度・VRAMは #2496 の共通harnessと固定速度入力で測る。現行モデルの速度基準は #2496 の値を使う（同じ条件を測り直さない）。harnessに足りない機能があれば、自分で別のharnessを作らずに親へ返す。
- ライセンス同意やアカウントが必要な重み（RF-DETR-2XLのRoboflow PML 1.0、Sapiens2など）は、ユーザーが同意・用意したうえで評価目的に使うと合意済み。必要になったら、何に同意し何を用意してほしいか（URL・必要な環境変数名など）を `needs_parent` で親へ返す。自分で同意・登録しない。認証や利用許諾を回避しない。本番採用できるかはライセンス上の制約として記録する。
- Sapiens2-5BのPyTorch FP32はローカルGPUで試行し、OOMを実行不能として必要メモリの見積もりと記録する。別GPUでは測らない。
- GPUはtraining queueへ `add` して使う（queue workerは親側で稼働中。自分で `start`・`serve` しない）。ローカルGPUは RTX 5060 Ti 16GB。
- 新しいpipeline経路（ECPose・FastHMR）は既存経路を残した明示選択の別recipeとして作り、暗黙のfallbackやcacheの混在をしない。経路の選び方の設計は #2491 が先に固め、#2487 のECPose経路も同じ形に合わせる。
- #1739 の新経路コード（PR #2449・#2469・#2461・#2470）は2026-11-24時点で未マージ。下流の新経路へ通す確認でこのコードが要る場合は、その時点のマージ状況を親へ確認する。
- 公式重みが取得できない・ライセンスで使えない場合、別サイズ・別データの重みへ無断で置き換えない。

## 土台

worktreeの土台は、#2495 のPR #2498（head dfbacf50）と #2496 のPR #2497（head 8dbafd61）を統合したcommit 9455acef8。両PRはまだdraftで変わりうる。土台のPRが更新されたら必要に応じて取り込み、両PRのマージ後はmainへ載せ替える（親が指示する）。土台のPRのコード自体は変更しない。不具合や不足を見つけたら親へ返す。
