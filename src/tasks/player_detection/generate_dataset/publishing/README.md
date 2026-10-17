# 最終player bboxの累積公開

`PlayerPublisher.publish(PublishRequest(plan, reviews))` は、[最終IDレビュー](../../annotation/player_pose/README.md)
の `FinalReview` だけを受け取り、planで固定した `player_detection/<version>/boxes` へ公開する。
`PlayerPublicationPolicy(image_storage="reuse_ball_jpeg", split="inherit_ball_source")` を
player/boxes productの `generation_policy` に指定する。実行時にsplitや閾値を決め直さない。

- 新規clipは最終player判定だけをbbox教師にする。非選手・重複は除外し、追加・修正boxは保持する。
  ID不明のレビューは拒否する。未確認・不完全・推定・未解決bboxのframeはignoreで保存する。
- 確認済み不在はinstance 0件の負例として保存する。全空clipもnative frame/JPEGを失わない。
  検出ゼロや全boxの画面外clipだけから負例を作らない。
- poseの生成・対象外判定はbbox公開を妨げない。各clipのsource splitは対応ball snapshotから継承する。
  既存同一sourceとsplitが異なる場合は停止する。
- base snapshotは[既存store固定API](../../data/snapshots/README.md)または前回の公開snapshot。
  既存clipのnative行・教師・semantic hash・splitを維持し、同じclipの追加/更新を拒否する。
  RGBは検証済みshardをhardlinkする。別filesystemへのcopyには暗黙に切り替えない。
- 公開先単位のlockと固定intentを持つstagingを使う。途中停止は同じintentだけが再開でき、
  検証後にrenameする。異なる所有者や不明なstagingを削除しない。完成snapshotを修復/上書きしない。
  rename後・receipt保存前の中断も同じ要求で共通 `PublicationReceipt` を回収できる。

通常のv2 readerに、宣言済みの `teacher_schema="player.annotation-supervision.v1"` と
`teacher_supervision` uint8列を追加する。値は `legacy=0, positive=1, negative=2, ignore=3`。
既存v2 clipは0を保ち従来の選別を維持する。同一clipのlegacy/明示mask混在は拒否する。
新規clipのignoreはreader設定でreviewed/located条件を緩めても学習に入らない。
mask・確認済み不在・instance数・実観測の整合性をreaderで検証する。

`load_publication(layout, receipt_ref)` はreceipt、固定要求、base、最終レビュー、snapshotの
membership/教師provenanceを合わせて検証する。追加学習が読むsnapshot/storeは既存の
[learning adapter](../../training/annotation/README.md)と同じ形式である。

`test_annotation_publication.py` は合成画像で累積公開・reader/collate・並行要求・途中停止を検証する。
`test_publication_loss.py` は公開した全空/混在batchを上流DINOのmatcher・classification/L1/GIoU lossへ
CPUで通し、有限値・負例の非ゼロ勾配・optimizer更新を検証する。DINOの初期化済みsubmoduleと
import可能な公式拡張が必要で、`PLAYER_DINO_REPOSITORY` で読取対象を明示できる。
CPUではCUDA専用denoising forwardを実行しない。全criterionの受入は、管理役が共有training queueの
allジョブで次を実行する（固定したDINO repository/拡張へのpathを環境に設定する）。

```bash
PLAYER_DINO_LOSS_DEVICE=cuda .venv/bin/python -m pytest -q --no-cov -n 0 \
  tests/integration/tasks/player_detection/test_publication_loss.py
```

これは合成prediction headの更新テストである。Swin本体を含む実画像の学習、追跡品質、ViTPoseの実出力、
実データのsource split対応・対象外率は、固定planの本番受入で別途確認する。
