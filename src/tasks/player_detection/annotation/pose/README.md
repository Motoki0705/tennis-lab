# 修正bboxからの追跡・ID・pose公開

注釈用検出の再実行はしない。`corrected.py` が採用bbox revisionと対応するball snapshotを照合し、
修正・手動追加した実観測をCLIP-ReID → poseなしStrongSORT++/AFLinkへ渡す。
実観測のbbox・観測ID・frame/source PTS・bbox hash・score由来を保存する。
手動bboxのscoreはmissingのままとし、model scoreを捏造しない。
[person_tracking](../../../person_tracking/README.md)の生成profileは変更しない。
trackerが確定trackへ出さなかった実観測も、独立したレビュー行として残す。

## 固定入力とAPI

共通の `AnnotationLayout.production()` と `ArtifactRef` を用いる。annotation rootや
公開versionは自由な実行時pathではなく、production planとrole rootから決定する。
数値閾値をここで決めず、`PosePolicy(presence_threshold, chunk_frames)` をplanの
player/pose `generation_policy` に固定する。存在率の分母はレビュー済みball対象frame、
分子は画面内に球の観測があるframe。閾値以下は理由付き対象外とする。
未レビューball frameがあるclipは生成対象として保持し、その理由を記録する。

1. `pin_input(layout, BboxInput(id_review, ball_snapshot))` でbbox採用revisionとball snapshotを固定する。
   `id_review` はbbox workerの `WorkOutcome.receipt`（`IdReviewBinding`）。版・観測ID・画像証拠を再検証する。
   canonical clip/source・media・geometry・native frame/PTSを照合し、JPEG shardを参照する。
2. `track_corrected(..., models=TrackingModels(clip_reid, aflink), extractor=..., aflink=...)`。
   poseなし `FeatureExtractor` を注入する。入力・重み・profileごとに追跡receiptを保存し、再開時も照合する。
3. `IdentityReview(tracking, decision, images, required_sheets)` を保存する。
   decision形式は [reviews.py](reviews.py) の `REVIEW_SCHEMA`、画像レビュー指示は [WORKER.md](WORKER.md)。
   全実観測を区間ごとにplayer/non_player/duplicate/unknownへ分類し、匿名 `player_N` を割り当てる。
   画像refとdecision、全観測coverage、同時ID衝突、重複の相手を検証する。
4. `finalize_review(layout, identity_review_ref)`。承認済みID判断を最終bboxへ戻し、入力revisionをsealする。
   非選手・重複の観測履歴は残し、教師から除く。追加bboxも維持する。
   教師maskと観測ID→匿名IDを `FinalReview` に固定し、ID確定後の中断は同じ入力で回復する。
5. `generate_pose(layout, final_review_ref, policy_ref=..., weights=..., model_factory=..., reuse=...)`。
   承認実観測だけにViTPoseを呼ぶ。非選手・重複・推定/補間・欠測には呼ばない。
   clip/media/native frame/PTS/JPEG bytes/bbox revision/最終labels/IDレビュー全体/選択/重みを照合する。
   同一identityのapproved結果のみ再利用できる。chunkの完了receiptがない部分だけを再実行する。
6. `PosePublisher(layout).publish(PosePublicationRequest(plan, ball_snapshot, results))`。
   planの `ball_detection/<version>/player_pose` へ不変snapshotを原子的に公開する。
   既存pose snapshotはplanのbaseから保持し、追加結果だけを渡す。ball snapshotの全clipが必要。
   `PublicationReceipt` を `PublishedPoseStore(layout, receipt_ref)` に渡して読む。

GPUモデルの構築・使用は必ず共有training queueのallジョブで行う。APIのモデル注入は
CPU合成テストにも使うため、実モデルのadmissionは呼出し元が担当する。
低位のcampaign実行入口もtask配下へ移設した。`generate-clip` は `corrected_inputs` に
clip ID→`BboxInput` refがない場合、モデルをロードする前に停止する。
CLI入口は `.venv/bin/python -m src.tasks.player_detection.annotation.pose`。
既存datasetの読込み・明示的移行に必要なmanifest検証は `dataset.py` / `expansion.py` にある。
#1088の公開・学習連携では、上記の固定plan・共通publication receiptを使用する。

## 状態と保存物

`needs_review` / unknown ID は保留で、最終教師・poseの公開を拒否する。
承認済み `FinalReview` だけでpose resultがなければ未生成である。
`PoseResult.status` は `generated` / `reused` / `not_applicable` を区別し、対象外理由を必須にする。
公開済みは共通publication receiptで判断する。readerは全依存refとreceiptの追加/既存membership、
採用receipt・plan・結果との一致を再検証し、理由付き対象外の場合だけ `None` を返す。
画像はballの既存JPEG shardを参照し、pose出力には複製しない。

bbox教師の累積公開は [generate_dataset/publishing](../../generate_dataset/publishing/README.md) が担当する。
存在率によるpose対象外でも、bboxの正例・確認済み負例を公開できる。

CPU合成テストは `tests/integration/tasks/player_detection/test_annotation_publication.py`。
修正/追加bbox、score由来、ID保留、推定除外、identity差異、chunk再開、ID seal後と
公開rename後/receipt前の障害を検証する。実モデルの画質・追跡品質・閾値の採用判断は本番受入で行う。
