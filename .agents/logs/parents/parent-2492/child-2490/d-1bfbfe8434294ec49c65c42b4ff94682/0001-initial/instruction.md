# #2490 補助担当 `rtmpose`（親 #2492）

担当issue #2490 の本文が仕様の正本。下の共通事項（親 #2492）が本文の「全69 clip」などより優先する。統合担当は primary で、比較表・可視化・報告は primary が行う。あなたの担当は次のモデル。

RTMPose-M と RTMPose-X（MMPose公式 https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose）。COCO17対応、または公式のbody keypoint定義からCOCO17へ厳密に変換できる公開重みを選ぶ。M/Xは可能な限り同じ学習データ系列・入力解像度にし、違えば明記する。

## やること

1. 公式の重み・入力仕様・COCO17への対応・ライセンス・ソースcommit・重みのsha256を固定する。評価利用と本番利用・再配布の可否を分けて記録する。
2. primaryが #2490 にコメントするadapterの置き場所・入出力の形に合わせて、出力を現行 `Pose2DResult` の (F,17,3) 契約へ変換するadapterを作る。primaryのコメントがまだなら、公式実装の導入・推論確認から進める。現行のconfidenceは生のheatmap peakで確率ではないので、新しいscoreを同じ閾値で無条件に使わない。
3. 入力は #2495 の固定疑似BBox（全モデル・全backendで同じ元画像・BBox・評価行）。モデル固有のaspect ratio・padding・resize・正規化だけ公式仕様に従い、元画像座標への逆変換を検証する。
4. #2496 の共通harnessで PT32/PT16/TRT16 の速度・VRAM・実行可否を測る（training queue経由）。使ったTTAとforward回数を記録する。TRT未対応・変換不可・parity不成立・OOMは条件ごとに理由を残し、暗黙のfallbackをしない。
5. #2495 の採点器で3 backendそれぞれの精度を採点する。PT32との出力差と疑似GTとの差を分けて出す。
6. 結果（全条件の実測値か実行不能の根拠、raw出力の場所、再現コマンド、hash）をcommitし、primaryが統合できる形で最終報告に返す。PRは作らず、branchとcommitを返す（primaryが統合する）。

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
