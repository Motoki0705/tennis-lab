# 素材の取得・共有台帳・クリップ準備

`Collection(layout, plan)` が不変planを登録し、`register()` → `acquire()` →
`select()` → `prepare()` を再開可能に実行する。画像によるプレー窓判定は
screeningが所有し、このモジュールは明示された `SelectionReceipt` だけを準備へ渡す。
均等時刻からの自動選定、短いクリップへのpadding、品質を下げて容量に合わせる処理は行わない。
統合CLI・実行枠・資源予算監視は[全体設計](../README.md)の対応モジュールが所有する。

## 入力と進捗

- `SourceCandidate` は配信元のcanonical channel/video ID、選定理由と証拠、取得先を持つ。
  ローカル素材はhashで固定した入力参照、YouTubeは取得metadataのID/channel IDを照合する。
- `PreparationConfig` は取得format、明示したH.264設定、実フレーム最小数、未準備素材の先行上限を持つ。
  原寸・PTS・表示時間を保持する。HDR、回転、非正方pixel、途中解像度変更は理由付きで拒否する。
- `SelectionReceipt` はsource hash、policy、画像選別の証拠参照、非重複media/target区間を固定する。
  最大クリップ数に不足する場合は理由を必須にする。0件は除外であり準備完了ではない。
- `summary()` は要求件数、確保channel、sourceの各状態、準備clip数、登録task数、未準備video数を区別する。
  取得失敗を成功に数えず、再取得には `acquire(..., retry=True)` を明示する。

## 共有台帳と再開

固定rootの `registry.sqlite3` をWALと書込みtransactionで共有する。
動画名やchannel表示名ではなく `(provider, video_id)`、元動画内容hash、clip区間identityを照合する。
新規channelの予約もcampaign横断で行う。既存データは収集前に
`CollectionLedger.import_history(source, evidence)` で読取棚卸しを登録する。
別campaignが確保済みのchannel/source、同内容の再アップロードは新規取得数に加えない。

取得/準備はsourceごとのOS file lockで多重起動を防ぐ。停止するとlockは解放される。
取得前に未準備素材の先行上限をtransaction内で確認し、満杯なら `Backpressure` を返す。
準備完了または除外までその枠を保持する。生きている別プロセスのlockを奪わない。

原動画は `media/sources/blobs/<sha256>` に1実体、取得receiptは
`media/sources/<source_id>/<preparation-config-hash>/acquisition.json` に保存する。
clipは `media/clips/<clip_id>/{media.mp4,manifest.json}` を一括renameで公開する。
内容や元動画区間の異なる既存clipを上書きしない。

取得・準備receiptを先に不変公開し、最後に台帳へ反映する。公開直後に停止しても、
再開時はreceiptを検証して台帳を補完する。準備途中で停止したときは検証済みclipを再利用する。
動画を再取得・再符号化して帳尻を合わせない。改変、消失、異なるplanは明示的な失敗になる。
preparedの登録時には選別区間・sourceの実PTS・clipのPTS/表示時間・hashを照合し、
clipと各taskを1 transactionで段階追加する。

汎用動画I/Oの正本は[media.py](../media.py)。旧Chat注釈の素材処理も同じ関数を参照し、
旧manifestの検証と画像ラベルだけを旧runtimeに残す。旧運用入口の廃止は統合タスクで行う。

検証: [test_collection.py](../../../../../tests/unit/tasks/base/annotation/test_collection.py)
で並行登録、過去素材、同内容、先行制限、実MP4のframe/PTS、公開前/後の障害、再試行を確認する。

クリップの通常選別は [screening](../screening/README.md) が担当する。
