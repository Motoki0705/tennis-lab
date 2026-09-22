# ChatGPT Project用のテニス動画アノテーション準備

YouTube URLから動画を取得し、前後の参考区間を含む最大15秒・500,000,000 bytes以下の
MP4と、Chatへ貼り付ける短いリクエスト本文を作る。対象は対象コートのプレーヤーと
プレー中のボールで、参考区間を含む添付動画の全フレームを処理する。
アノテーションの正本・座標・補間・可視化・返却形式は
[PROTOCOL.md](resources/PROTOCOL.md)、機械契約は
[runtime/contracts.py](runtime/contracts.py)を参照する。JSON Schemaは内部の版識別用に生成し、Chatへは渡さない。

## 実行

repo/worktreeのrootから:

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.prepare \
  'source.url="https://www.youtube.com/watch?v=VIDEO_ID"'

# 複数URL。URL内の=をHydraが解釈しないよう各値をdouble quoteで囲む。
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.prepare \
  'source.urls=["https://www.youtube.com/watch?v=VIDEO_ID_1","https://www.youtube.com/watch?v=VIDEO_ID_2"]'

# 前後の重複区間込みで20秒。注釈は重複区間を含む20秒全体を対象にする。
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.prepare \
  'source.url="https://www.youtube.com/watch?v=VIDEO_ID"' \
  clip.duration_seconds=20 clip.context_seconds=1

# 保存済み動画でオフライン確認。パスはpaths.data_rootからの相対パス。
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.prepare \
  source.local_video=samples/tennis_clip.mp4
```

全設定は[configs/prepare.yaml](configs/prepare.yaml)。CLI overrideも同じ設定を使用する。
`source.url`、`source.urls`、`source.local_video`は排他的。HTTPSのYouTube動画URLを
受け付け、playlist取得は行わない。単一URLと複数URLの設定例:

```yaml
# 1動画
source:
  url: "https://www.youtube.com/watch?v=VIDEO_ID"
  urls: []

# 複数動画（入力順を維持）
source:
  url: null
  urls:
    - "https://www.youtube.com/watch?v=VIDEO_ID_1"
    - "https://www.youtube.com/watch?v=VIDEO_ID_2"
batch:
  download_workers: 2
```

複数URLではダウンロードだけを`batch.download_workers`本まで並行し、取得済み動画の
エンコードは1動画ずつ行う。ダウンロードとエンコードは重なるが、実効速度は回線・配信元・
CPUに依存する。1件が失敗しても残りを処理し、入力順の結果とエラーを`batches/*/batch.json`へ
保存して全体を失敗として終了する。同じYouTube動画IDの重複は設定エラーにする。

YouTubeの取得環境に応じて`source.js_runtimes=node`等を明示指定できる。
source.format_selectorの既定値は
`bv[ext=mp4][vcodec^=avc1][dynamic_range=SDR][height<=1080]`。
音声なしのMP4/H.264、SDR、最大1080pに固定し、元のFPSは制限しない。
条件に合わない形式へ静かにfallbackしない。注釈用出力は8-bit H.264のため、
PQ/HLGのHDRを暗いSDRとして誤表示しないようHDR入力は明示的に拒否する。
元解像度を保持し、CPU libx264でCRF18のMP4に変換する。音声は保存クリップから除外する。
B-frameを無効にし、可変FPSでも末尾フレームの表示時間まで保存する。
回転メタデータ・非正方画素・奇数解像度・不正なPTSは自動補正せずエラーにする。

worktreeから共通データと出力を使う場合、`paths.data_root`と`paths.output_root`に元repo配下の
絶対パスを指定する。local_video/output_directoryは対応rootからの相対パスを指定する。
ダウンロード用にネットワークとyt-dlp、処理用にrepoの通常Python依存が必要。GPUは使用しない。

## 出力とChatへの添付

```text
outputs/chat_annotation/             # output_directoryで変更可能
  sources/                         # 保存した元動画、取得metadata、ハッシュ記録
  project_kits/
    PROJECT_INSTRUCTIONS.txt       # Project instructions用
    REQUEST.txt                    # 短い要求・JSON例・最小限の入力一覧
  videos/<source-video-name>/      # 保存したソース動画のファイル名（拡張子なし）
    <source-id>__<run-hash>__<clip-id>.mp4
    ...                            # 同じソースのクリップを並べる
  _preparation/<source-id>/<run-hash>/
    prepared.json                  # ローカルでの再実行・整合性確認用
    ready/<clip-id>.json
    clips/<clip-id>/clip_manifest.json
```

1. Projectを使う場合は`project_kits/PROJECT_INSTRUCTIONS.txt`をProject instructionsへ貼る。
2. クリップごとのChatでgpt-6-astraを選び、`videos/<source-video-name>/`からクリップ1本だけを添付する。
3. `project_kits/REQUEST.txt`の全文をプロンプトとして貼り付ける。
4. 返却ZIP内の注釈JSONと重畳動画を確認する。部分完了もJSONに明示される。

REQUESTには要求と短いJSON例、各動画のファイル名・幅・高さ・総フレーム数・補間上限だけを
埋め込む。同じREQUESTを準備済み動画に共通で使い、添付ファイル名に一致する行を選ぶ。
動画はファイル名を変更せず添付する。別のJSONやPROTOCOLファイルの添付は不要。
元動画の出典・ハッシュ・全PTS・フレーム対応はローカルのmanifestに保持する。
Pythonコードは配布せず、実装はgpt-6-astraに任せる。PROTOCOLは要求文書とJSON例の
唯一の保守元で、REQUESTへ組み込む。実装方法・ライブラリ・作業順序・応答の行数は指定しない。

`_preparation/`のmanifestと完成マーカーはローカルでの再実行検証用で、Chatには渡さない。
`project_kits/`には常に2つのテキストを生成し、ハッシュ付きキットディレクトリは作らない。
複数動画・複数URLの準備では、それまでに公開済みの入力情報もREQUESTへ残す。
要求の版が異なる既存動画と混在する場合は明示的に失敗するため、新しいoutput_directoryを使う。
実際のChatでの動画処理・コード実行・ダウンロードは利用環境で確認が必要。
機械検証は注釈の意味的精度を保証しない。

## 分割・再実行

既定では1動画につき最大5クリップを採取する。通常分割の候補が5本以下なら全候補を使い、
超える場合は全時間を5枠に等分して各枠の中央付近から担当区間を選ぶ。
`sampling.max_clips_per_video: null`にすると従来どおり全フレームを順次担当する。
方式は`sampling.strategy: uniform_midpoints`のみを受け付ける。`prepared.json`には候補数、
依頼範囲と実際の選択範囲、選択フレーム数、`full`/`sampled`のcoverageを記録する。

表示順frame indexと元PTS/time_baseを保存し、CFR化・間引き・縮小を行わない。
フレーム境界に丸めるため長さは指定値以下になり、末尾は短くなる。
採取モードでは出力クリップ内の全フレームが注釈対象であり、クリップ外の非選択フレームをnegativeとは扱わない。
容量を超えた枠は中心を保って短縮し、1枠を複数クリップへ増やさない。全量モードでは担当範囲が
半開区間で全元フレームを一度ずつ覆い、前後の参考区間だけが重複する。容量超過時の二分も
全量モードだけで行う。注釈JSONには重複区間も含め、結合時はローカルmanifestの
source_frame_index/is_targetで所有範囲を識別できる。1フレーム＋文脈でも上限を超える場合は明示的に失敗する。
`-fs`による打ち切りや品質変更で成功扱いしない。

キット、元動画、設定のハッシュごとに出力を分離する。正常な完成物は再実行で検証して再利用し、
一時出力は完成物として扱わない。変更・破損した完成物はエラーにするため、新しいoutput_directoryを
指定するか、対象を確認して除去してから再実行する。自動で人の結果を上書きしない。
元動画は保持するため、取得と変換の両方に必要なディスク容量を確保する。

## 実装とテスト

- configuration/prepare/preparation: 厳密な設定、既存YouTube取得API、容量検査と公開。
- kit/prompt: 注釈型で版を識別し、要求文書と最小限の動画情報からREQUESTを生成。
- runtime: repo内で使用する型、動画I/O、補完、検証、描画、ZIP処理。Chatには配布しない。
- resources: Project指示・成果物要求の唯一の保守元。

```bash
.venv/bin/python -m pytest tests/unit/tennis_scene/chat_annotation tests/e2e/tennis_scene/test_chat_annotation.py
.venv/bin/python -m ruff check src/tennis_scene/chat_annotation tests/unit/tennis_scene/chat_annotation tests/e2e/tennis_scene/test_chat_annotation.py
.venv/bin/python -m mypy src/tennis_scene/chat_annotation
```

YouTube取得はテストではmockにし、動画分割と配布キットは実際にエンコード・デコードする。
E2Eは動画とREQUEST本文の例から合成注釈を作り、ローカルmanifestを使って返却ZIPを検証する。
CFR/VFRの全フレーム・表示時間、参考区間の描画、completed/partialの2ファイル構成、無効入力の拒否を確認する。
GPTが独自に作るコードやChat上での注釈結果そのものは自動テストの対象外。

ローカルで返却注釈を検証する場合は、元のmanifestを指定する（すべて絶対パス）。

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.annotate validate \
  --manifest /absolute/path/clip_manifest.json \
  --annotations /absolute/path/annotation_CLIP.json \
  --report /absolute/path/report.json
```

同CLIの`finalize`は`--video`、`--manifest`、`--annotations`、`--output`を受け取り、
有効なcompleted/partialだけを2ファイルのZIPとして生成する。構造・入力エラーや
動画生成失敗ではZIPを公開せず非0終了する。既存の出力ディレクトリは上書きしない。
