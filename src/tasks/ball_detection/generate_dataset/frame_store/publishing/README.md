# Ball frame storeの増分公開

`BallPublisher.publish(PublishRequest(...))`は、seal済みのballレビューを、
`BallPublisher.publish(VideoPublishRequest(...))`は、登録済み元動画の完成v3注釈を
`data/ball_detection/<version>/frames/`へ新しい不変snapshotとして公開する。
`ProductPlan`の`dataset_task=ball_detection`、`component=frames`を要求する。
別componentのposeは同じversion配下へ独立に公開できる。
readerにはversion親ではなく、snapshotが指定するframe storeを渡す。
正式CLI・資源管理付きstageへの接続はcomposition rootが担当する。

data-rootのファイル参照を作る`data_reference`は、共有APIの
[`src.tasks.base.annotation.artifacts`](../../../../base/annotation/artifacts.py)から
直接importする。`snapshot.py`はこの共通APIを再exportしない。

## 元動画v3の公開方針

元の[`video_ball_annotation.v3`](../../../annotation/video/README.md)を正式readerで直接読み、
登録済みの独立した`VideoAnnotationInput`と照合する。sceneの三角測量・平滑化した座標を
GPTの元注釈へ戻さない。新しい独立QA・observed_onlyの採用gateは加えない。

- 対象はcameraが向くコートでプレーする生きた1球、`target_semantics="active_ball"`。
  手持ち球・予備球・隣接コートの球を網羅する教師ではない。
- 位置利用可の全点（observed / interpolated / estimated / occlusion_estimated）を
  座標・heatmapの正例とする。**観測と推定は同じ重み1という初期設計案**を明示的に採用し、
  `position_weighting="uniform_observed_and_estimated"`とsnapshotの`position_weights`へ固定する。
  低重み化、追加の補間時間制限、observedへの由来の付替えを行わない。
- `no_active_ball`と`out_of_frame`はactive_ballだけのheatmap負例。座標教師はない。
  最終`unknown`と座標なしの遮蔽は両教師maskを無効にする。状態・遮蔽・点種・support・notesを
  元v3のままsnapshot metadataに保持し、不在とunknownを合流させない。
- planの`publish_dataset=True`と、登録した全camera・全関連clipの完成を要求する。
  最終unknownは完成に含めるが、未処理が1frameでもあれば公開を拒否する。部分集合の公開も拒否する。
  公開しなくても保存済み注釈は残る。
- `image_storage="jpeg", jpeg_quality=90`を要求する。登録時のnative Q90 JPEG archiveを
  bytes・offset・frame別hashを照合してhardlinkし、再圧縮・resize・paddingしない。
  storeの`pts`はrebased clip時刻ではなく元v3の実整数PTS、frameの元番号は保持したv3にある。
  元解像度、time_base、native FPS、camera、元動画identityも保存する。
- `recording_source_id(recording_id)`（登録時の収録channel IDと同じ）をgroup/source IDとする。
  全cameraと関連clipは同じgroup/splitになる。`source_splits`または`source_partition`を明示し、
  既存baseのholdoutを含むsplit変更は拒否する。local planのpartition件数は動画本数ではなく
  **新規収録group数**である。少数収録で比率を正確に表せなければ明示splitを使う。
- 注釈schemaのv3、frame store schemaの`ball_detection_frames.v4`、
  `ProductPlan.dataset_version`（任意の新snapshot版名）は別々の版である。
  旧storeの教師対象は自動的にactive_ballと解釈しない。v3公開のbaseにはactive_ball snapshotだけを
  認め、旧v1〜v3の汎用教師との混在はtarget semantics不一致として拒否する。旧データ移行は別工程。
- 同じrequestの再送は同じreceiptを返す。別requestで既存clipや同じ元動画/cameraの重なる
  frame区間を追加することは、修正版の注釈であっても拒否する。修正は旧注釈revisionと旧snapshotを
  保存し、新revisionから新しい系列を明示して公開する。既存教師の上書き・自動置換はしない。
- 公開先はrepo内の`data/ball_detection/<dataset_version>/frames/`のみ。
  外部配布・元動画の削除・学習submit・モデル採用を行わない。CLIの公開選択との配線はcomposition rootが担う。

```python
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    SavedVideoAnnotation, VideoPublishRequest,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import BallPublisher

# plan_ref: configure_local_videosが保存したpublish_dataset=Trueのlocal plan
# output: ProductPlan(ball_detection/frames、新dataset_version、上記policy参照、任意のactive_ball base)
# saved: 全登録contextと各cameraの保存済みv3 annotation revisionのArtifactRef
request = VideoPublishRequest(
    plan=plan_ref, output=output,
    annotations=tuple(sorted(
        (SavedVideoAnnotation(context=context, annotation=revision) for context, revision in saved),
        key=lambda entry: entry.context.digest,
    )),
)
receipt_ref = BallPublisher(layout).publish(request)
```

既存のintent・staging・rename・receipt・registryの復旧手順をそのまま通す。
共通`PublicationReceipt.annotations`へ保存v3 revisionを記録し、旧レビュー向け`adoptions`は空にする。
snapshotは元v3全体を内包するため学習時にlive注釈を必要とせず、来歴参照は公開保護へ登録する。
専用学習readerと出力maskは[data README](../../../data/README.md#active_ballの学習reader)を参照。

## 保存済みv2の移行公開

`BallPublisher.publish(MigrationPublishRequest(...))` は、固定した移行対象集合を
同じintent・staging・rename・receipt・registry経路で公開する。
新規注釈campaign、worker、実行資源の予算、学習jobを作らない。
`output` は上記と同じactive_ballの新version・明示split・native Q90方針を指定する。

各 `MigratedVideoAnnotation` は、保存した独立 `expected_input`、元v2 `original`、
変換したv3 `annotation`、taskの `MigrationRecord`、prepared画像の `manifest`、
保持した元動画 `source_media` のhash付き参照を必須にする。
requestの全入力はexpected_inputのdigest順で一度ずつ列挙する。
公開時には元v2からconverterを再実行してv3/移行記録を照合し、元動画bytesと実復号PTS、
画像manifestのcamera・収録group・frame/PTS・寸法・Q90 archiveを検証する。
元版・移行記録・入力参照はsnapshotの由来と公開保護に残す。

旧storeの汎用教師はbaseへ混ぜない。移行の呼出側は、旧snapshotの元収録group/splitを
独立に照合して同じsplitを指定し、変換できない対象と理由を別に保存する。
公開要求自体が選択集合であり、未列挙の旧データまで移行済みとは扱わない。
同一要求の再送・中断復旧・既存frame区間の追加拒否は通常のv3公開と同じ。

## 入力と教師

- 旧レビューrequestは不変campaign plan、product名、digest順の採用receiptを固定する。
- 各receiptは同じplanの現在revisionにsealされ、実画像根拠・方針・教師maskを再検証できなければならない。
- `BallPublicationPolicy`は画像保存方式とsplitの決め方を固定する。`source_splits`によるcanonical source IDの列挙か、`source_partition`によるIDとseedのhash順での配分を明示する。両方の併用、既存sourceのsplit変更を拒否する。
- partition方式は収集前にseedとtrain/validation/testの整数比を固定し、要求動画数が揃った公開時に新規sourceだけを配分する。例えば100動画と8:1:1なら80/10/10動画で、同一動画のclipは同じsplitになる。途中の候補差替えでplanを変更する必要はない。ラベル・モデル誤差は配分に使わず、比率を正確に表せない件数や部分公開は拒否する。既存snapshotのsplitは維持する。
- 新規画像は全native frameを元PTS・元解像度でJPEG化する。padding、追加resize、非プレーframeの補充を行わない。
  JPEGで準備されたclipは、保存済みQ90 bytes・offset・hashを検証し、そのarchiveを公開先へhardlinkする。
  推論・GPT確認で使った画像を再圧縮・二重保存せず、同じ画像を教師に用いる。
  元動画が削除済みでも、準備manifestのJPEG参照だけで公開できる。
  `image_storage=jpeg`はJPEG本体を保存し、`video_q90`はprepared MP4とQ90 JPEGのframe別hashを保存する。
  後者はJPEG本体をディスクへ展開せず、同じ入力bytesを学習時に再現する。
  保存と読出しの契約は[data README](../../../data/README.md)を参照。
- `annotated`、観測座標、遮蔽、補間、不在、未解決を保持し、採用maskをstore v2へ書き込む。保存・学習上の意味は[data README](../../../data/README.md)を参照。
- 全frameのsource対応、notes、詳細boundary、画像根拠、予測版は元manifest/revision/adoptionへのhash付き参照で保持する。storeの単一event列にhitとbounceを同時に表せない場合は`unlabeled`とし、`segment_break`と元revisionの両eventを保持する。`none`を捏造しない。

## 既存storeの導入

既存v1は`freeze_legacy_store(layout, directory, groups)`でindex・metadata・全shardをhash固定する。
`LegacySourceGroup`は既存`source/group_id`とcanonical source IDの対応を全groupについて要求する。
固定validation/testにも同じcanonical IDを使い、旧名と新名の違いによるsource漏洩を防ぐ。
旧clip IDは`content_digest(["legacy_ball_clip", source, store_clip_id])`として不変のmembership IDを得る。
chat_annotationの全clipには`clip_ranges`で`LegacyClipRange`（旧store名と元動画のframe区間）を明示し、canonical sourceと区間から得るcanonical clip IDもsnapshotに固定する。累積公開でもこの対応を保持し、旧名で導入したclipの再採用を拒否する。
元のstore clip名・row順・split・画素・ラベルは変更しない。

この導入は既存データの書換えや新しい品質承認ではない。
公開後は`PublicationReceipt.snapshot`を次のplanの`base_snapshot`へ渡し、v2をlegacyとして再導入しない。
既存教師を変更する追加要求と、既存clipの重複追加は拒否する。

## 公開・復旧

1. base snapshot、seal、splitを検証し、共有registryへ公開intentと出力先の予約を保存する。
2. 同じ出力先を`artifact_lock`で排他し、同じfilesystemのstagingへ既存JPEG/video payloadをhardlinkする。別filesystemへの暗黙copyはしない。
3. 新規clipのJPEGを準備済みarchiveから共有するか、動画保存の明示policyに沿ってencodeし、
   既存列のprefix・clip metadata・意味的hash・教師maskを保持して新indexを作る。
4. metadataと`BallSnapshot`を保存し、全ファイルhash・membership・採用ラベルとの一致を検証する。
5. directoryを原子的にrenameし、`PublicationReceipt`を書き、registryへ完了を登録する。

同じrequestの再実行は同じreceiptを返す。完成したstagingやrename済みsnapshotは再encodeしない。
shard/index作成途中なら、そのintentが所有する未完成stagingだけを再構築する。
再構築の清掃中も所有者ファイルを保持し、清掃自体が中断されても同じintentで再開する。
rename後・receipt前、receipt後・DB登録前も同じ出力を検証して復旧する。
異なるintent、改変された完成snapshot、所有者不明のstagingはエラーとし、公開済みデータを削除しない。

`BallSnapshot`はindex・metadata・各shard、clip/source membership、split、意味的hashを固定する。
snapshotの`shard`とwindow manifestの`rgb_shard_sha256`はclipの物理payloadを指す。
動画保存ではMP4の参照/hashで、形式と再生成JPEGの同一性はhash固定されたstore metadataにも記録する。
`load_snapshot`はこれらを検証して`BallFrameStore`を返す。hardlinkは別名での不変共有であり、
in-place書換えから保護する仕組みではないため、consumerでもhashを検証する。
registryのintent/receiptはbase、採用根拠、公開snapshotへの参照を保持する。GCはこの参照を保護対象として辿る必要がある。
学習への接続契約は共通の`PublicationReceipt`で、追加clip IDと再利用clip IDを分ける。
