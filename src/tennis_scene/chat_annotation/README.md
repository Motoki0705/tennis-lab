# ChatGPT Project用のテニス動画アノテーション準備

YouTube URLから動画を取得し、前後の参考区間を含む最大15秒・500,000,000 bytes以下の
MP4と元動画とのフレーム対応manifestを含む、動画とChatへ貼り付けるリクエスト本文を作る。
アノテーションの正本・座標・役割・補間・可視化・返却形式は
[PROTOCOL.md](resources/PROTOCOL.md)、機械契約は
[runtime/contracts.py](runtime/contracts.py)を参照する。JSON Schemaはこの型から生成する。

## 実行

repo/worktreeのrootから:

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.prepare \
  'source.url="https://www.youtube.com/watch?v=VIDEO_ID"'

# 複数URL。URL内の=をHydraが解釈しないよう各値をdouble quoteで囲む。
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.prepare \
  'source.urls=["https://www.youtube.com/watch?v=VIDEO_ID_1","https://www.youtube.com/watch?v=VIDEO_ID_2"]'

# 長さは参考区間込み。20秒中、通常18秒を担当し前後各1秒を参考にする。
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
    REQUEST.txt                    # 要求・スキーマ・定義・動画情報を含む全文プロンプト
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
4. 返却ZIPの検証結果と重畳動画を確認する。

要求・JSON Schema・CourtKP20定義・キット版・各動画の元動画情報・フレーム対応はREQUEST本文へ
埋め込む。ファイル名とSHA-256で今回の動画を識別し、同じREQUESTを準備済み動画に共通で使う。
フレーム対応は等しい表示時間の連続区間で表し、CFR/VFRとも時刻情報を失わず本文量を抑える。
動画はファイル名を変更せず添付する。別のJSONやPROTOCOLファイルの添付は不要。
Pythonコードは配布せず、実装はgpt-6-astraに任せる。PROTOCOLは要求文書の保守元としてrepo内に置き、
REQUESTへ組み込む。実装方法・ライブラリ・作業順序を指定しない。

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
採取モードでは選ばれた区間だけが注釈対象であり、非選択フレームをnegativeとは扱わない。
容量を超えた枠は中心を保って短縮し、1枠を複数クリップへ増やさない。全量モードでは担当範囲が
半開区間で全元フレームを一度ずつ覆い、前後の参考区間だけが重複する。容量超過時の二分も
全量モードだけで行う。1フレーム＋文脈でも上限を超える場合は明示的に失敗する。
`-fs`による打ち切りや品質変更で成功扱いしない。

キット、元動画、設定のハッシュごとに出力を分離する。正常な完成物は再実行で検証して再利用し、
一時出力は完成物として扱わない。変更・破損した完成物はエラーにするため、新しいoutput_directoryを
指定するか、対象を確認して除去してから再実行する。自動で人の結果を上書きしない。
元動画は保持するため、取得と変換の両方に必要なディスク容量を確保する。

## 実装とテスト

- configuration/prepare/preparation: 厳密な設定、既存YouTube取得API、容量検査と公開。
- kit/prompt: CourtKP20定義・注釈型・要求文書と動画情報から、自己完結するREQUESTを生成。
- runtime: repo内で使用する型、動画I/O、補完、検証、描画、ZIP処理。Chatには配布しない。
- resources: Project指示・成果物要求の唯一の保守元。

```bash
.venv/bin/python -m pytest tests/unit/tennis_scene/chat_annotation tests/e2e/tennis_scene/test_chat_annotation.py
.venv/bin/python -m ruff check src/tennis_scene/chat_annotation tests/unit/tennis_scene/chat_annotation tests/e2e/tennis_scene/test_chat_annotation.py
.venv/bin/python -m mypy src/tennis_scene/chat_annotation
```

YouTube取得はテストではmockにし、動画分割と配布キットは実際にエンコード・デコードする。
E2Eは動画とREQUEST本文だけを独立した場所へ移し、本文から定義を復元してローカルruntimeによる返却ZIPを確認する。
GPTが独自に作るコードやChat上での注釈結果そのものは自動テストの対象外。
