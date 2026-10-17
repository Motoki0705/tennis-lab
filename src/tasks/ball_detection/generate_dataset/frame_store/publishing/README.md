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
- `BallPublicationPolicy`はJPEG品質とcanonical source IDごとのsplitを明示する。新sourceの割当てを推測せず、既存sourceのsplit変更も拒否する。
- 新規画像は全native frameを元PTS・元解像度でJPEG化する。padding、追加resize、非プレーframeの補充を行わない。
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
2. 同じ出力先を`artifact_lock`で排他し、同じfilesystemのstagingへ既存JPEG shardをhardlinkする。別filesystemへの暗黙copyはしない。
3. 新規clipだけをencodeし、既存列のprefix・clip metadata・意味的hash・教師maskを保持して新indexを作る。
4. metadataと`BallSnapshot`を保存し、全ファイルhash・membership・採用ラベルとの一致を検証する。
5. directoryを原子的にrenameし、`PublicationReceipt`を書き、registryへ完了を登録する。

同じrequestの再実行は同じreceiptを返す。完成したstagingやrename済みsnapshotは再encodeしない。
shard/index作成途中なら、そのintentが所有する未完成stagingだけを再構築する。
再構築の清掃中も所有者ファイルを保持し、清掃自体が中断されても同じintentで再開する。
rename後・receipt前、receipt後・DB登録前も同じ出力を検証して復旧する。
異なるintent、改変された完成snapshot、所有者不明のstagingはエラーとし、公開済みデータを削除しない。

`BallSnapshot`はindex・metadata・各shard、clip/source membership、split、意味的hashを固定する。
`load_snapshot`はこれらを検証して`BallFrameStore`を返す。hardlinkは別名での不変共有であり、
in-place書換えから保護する仕組みではないため、consumerでもhashを検証する。
registryのintent/receiptはbase、採用根拠、公開snapshotへの参照を保持する。GCはこの参照を保護対象として辿る必要がある。
学習への接続契約は共通の`PublicationReceipt`で、追加clip IDと再利用clip IDを分ける。
