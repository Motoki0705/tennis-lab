# 旧chat annotationの移行reader

`LegacyReader` は旧 `tennis_chat_preparation.v2` と `tennis_chat_clip.v1` を読み、
検証済みの閉じたファイル集合を[共通移行処理](../../base/annotation/storage/migration/README.md)へ渡す。
通常の収集・注釈フローとは別の入口で、旧データを新しい教師として採用する処理ではない。

`LegacySelection` に次を明示する。root全体の自動探索や、未対応ファイルの黙ったスキップは行わない。

- `prepared_runs`: `_preparation/<source>/<run>` の完成したrun。summaryに列挙された全clipを含める。
- `sources`: 各sourceの `source_record.json` と元動画。source ID・ファイル名・全metadata・hash/sizeを照合する。
- `annotations`: 残すprocessed ball/player JSON。選択runのclip ID・全frame順・寸法・座標・statusを旧validatorで検証する。
- `raw_zips`: 残すraw ZIP。旧ZIP検査と注釈validatorを通し、全memberが参照するclipを選択runに含める。
- `max_clips` / `max_json_bytes`: 読取範囲の上限。

移行単位を完成runにすることで、`prepared.json` の対象範囲・選別slot・coverageと、
各ready receipt・clip manifest・動画をまとめて検証できる。source-localのframe区間IDと、
注釈・datasetが使う動画filename stemのclip IDを両方保持する。`videos` と `done` の配置も
そのまま保持し、両方に同じclipがある場合は拒否する。元動画はsource単位の共有依存として扱う。
ZIPは複数clipを含んでも1つのhash artifactで保持し、分割や内容の書換えをしない。

```python
from src.tasks.annotation.migration import LegacyReader, LegacySelection, LegacySource
from src.tasks.base.annotation.storage.migration import MigrationStore, dry_run

# layout / legacy_root / 相対path / 各上限は呼出側で明示する。
reader = LegacyReader(LegacySelection(
    prepared_runs=(run_relative_path,),
    sources=(LegacySource(record=source_record_relative_path, video=source_video_relative_path),),
    annotations=processed_relative_paths,
    raw_zips=raw_zip_relative_paths,
    max_clips=clip_limit,
    max_json_bytes=json_size_limit,
))
plan = dry_run(layout, reader, root_id=import_identity, source_root=legacy_root,
               mode="in_place", max_files=file_limit)
# 実行は資源受付後に明示する。別rootのコピーでは mode="copy" を指定する。
receipt = MigrationStore(layout).apply(plan, reader, max_copy_bytes=copy_payload_limit,
                                      max_files=file_limit)
```

prepared runにない元素材、Project kit、任意のQA成果物、local-agent campaignの絶対path設定、
別rootにあるdataset/pose storeはこのselectionの対象ではない。これらを含めて移行済みとは
扱わず、元の場所に残す。特にcampaign設定のpathを書き換えて再開可能と見せたり、pose
manifest内のball store参照を自動変更したりしない。追加形式は別readerで検証して扱う。

CPUの合成動画では、元動画・2つのclip・ball/player JSON・複数clip入りZIPを実際の旧writerで
生成し、コピー後に既存ball/player dataset collectorから読めることを検証する。実動画、GPU、
既存運用データの移動はこのテストでは行わない。
