# Ball frame storeの増分公開

`BallPublisher.publish(PublishRequest(...))`は、seal済みのballレビューを
`data/ball_detection/<version>/frames/`へ新しい不変snapshotとして公開する。
`ProductPlan`の`dataset_task=ball_detection`、`component=frames`を要求する。
別componentのposeは同じversion配下へ独立に公開できる。
readerにはversion親ではなく、snapshotが指定するframe storeを渡す。
正式CLI・資源管理付きstageへの接続はcomposition rootが担当する。

## 入力と教師

- requestは不変campaign plan、product名、digest順の採用receiptを固定する。
- 各receiptは同じplanの現在revisionにsealされ、実画像根拠・方針・教師maskを再検証できなければならない。
- `BallPublicationPolicy`は画像保存方式とsplitの決め方を固定する。`source_splits`によるcanonical source IDの列挙か、`source_partition`によるIDとseedのhash順での配分を明示する。両方の併用、既存sourceのsplit変更を拒否する。
- partition方式は収集前にseedとtrain/validation/testの整数比を固定し、要求動画数が揃った公開時に新規sourceだけを配分する。例えば100動画と8:1:1なら80/10/10動画で、同一動画のclipは同じsplitになる。途中の候補差替えでplanを変更する必要はない。ラベル・モデル誤差は配分に使わず、比率を正確に表せない件数や部分公開は拒否する。既存snapshotのsplitは維持する。
- 新規画像は全native frameを元PTS・元解像度でJPEG化する。padding、追加resize、非プレーframeの補充を行わない。
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
元のstore clip名・row順・split・画素・ラベルは変更しない。

この導入は既存データの書換えや新しい品質承認ではない。
公開後は`PublicationReceipt.snapshot`を次のplanの`base_snapshot`へ渡し、v2をlegacyとして再導入しない。
既存教師を変更する追加要求と、既存clipの重複追加は拒否する。

## 公開・復旧

1. base snapshot、seal、splitを検証し、共有registryへ公開intentと出力先の予約を保存する。
2. 同じ出力先を`artifact_lock`で排他し、同じfilesystemのstagingへ既存JPEG/video payloadをhardlinkする。別filesystemへの暗黙copyはしない。
3. 新規clipだけをencodeし、既存列のprefix・clip metadata・意味的hash・教師maskを保持して新indexを作る。
4. metadataと`BallSnapshot`を保存し、全ファイルhash・membership・採用ラベルとの一致を検証する。
5. directoryを原子的にrenameし、`PublicationReceipt`を書き、registryへ完了を登録する。

同じrequestの再実行は同じreceiptを返す。完成したstagingやrename済みsnapshotは再encodeしない。
shard/index作成途中なら、そのintentが所有する未完成stagingだけを再構築する。
rename後・receipt前、receipt後・DB登録前も同じ出力を検証して復旧する。
異なるintent、改変された完成snapshot、所有者不明のstagingはエラーとし、公開済みデータを削除しない。

`BallSnapshot`はindex・metadata・各shard、clip/source membership、split、意味的hashを固定する。
snapshotの`shard`とwindow manifestの`rgb_shard_sha256`はclipの物理payloadを指す。
動画保存ではMP4の参照/hashで、形式と再生成JPEGの同一性はhash固定されたstore metadataにも記録する。
`load_snapshot`はこれらを検証して`BallFrameStore`を返す。hardlinkは別名での不変共有であり、
in-place書換えから保護する仕組みではないため、consumerでもhashを検証する。
registryのintent/receiptはbase、採用根拠、公開snapshotへの参照を保持する。GCはこの参照を保護対象として辿る必要がある。
学習への接続契約は共通の`PublicationReceipt`で、追加clip IDと再利用clip IDを分ける。
