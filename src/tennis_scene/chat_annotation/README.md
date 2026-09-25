# ChatGPT Project用のテニス動画アノテーション準備

YouTube URLから動画を取得し、前後の参考区間を含む最大15秒・500,000,000 bytes以下の
MP4と、Chatへ貼り付ける短いリクエスト本文を作る。対象は対象コートのプレーヤーと
プレー中のボールで、参考区間を含む添付動画の全フレームを処理する。
アノテーションの正本・座標・補間・可視化・返却形式は
[PROTOCOL.md](resources/PROTOCOL.md)、機械契約は
[runtime/contracts.py](runtime/contracts.py)を参照する。対象別REQUESTには機械契約から生成したJSON Schemaを含める。

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
    ball_detection/
      PROJECT_INSTRUCTIONS.txt     # ボール注釈Project用
      REQUEST.txt                  # ボール用依頼文とJSON Schema
    player_detection/
      PROJECT_INSTRUCTIONS.txt     # プレーヤー注釈Project用
      REQUEST.txt                  # プレーヤー用依頼文とJSON Schema
  videos/<source-video-name>/      # 保存したソース動画のファイル名（拡張子なし）
    <source-id>__<run-hash>__<clip-id>.mp4
    ...                            # 同じソースのクリップを並べる
  _preparation/<source-id>/<run-hash>/
    prepared.json                  # ローカルでの再実行・整合性確認用
    ready/<clip-id>.json
    clips/<clip-id>/clip_manifest.json
```

1. 注釈対象に応じて`project_kits/ball_detection/`または`project_kits/player_detection/`を選び、その`PROJECT_INSTRUCTIONS.txt`をProject instructionsへ貼る。
2. 対象別のChatでgpt-6-astraを選び、`videos/<source-video-name>/`から注釈するクリップを添付する（複数本可）。
3. 選んだディレクトリの`REQUEST.txt`全文をプロンプトとして貼り付ける。
4. 動画ごとの注釈JSONをまとめたZIPをMCPへ提出させ、保存結果を確認する。overlay動画はChat上で個別に確認する。詳細な提出契約は対象別REQUESTを参照。

各REQUESTには対象別の要求と専用JSON Schemaを含める。ボール用JSONはボール情報だけ、プレーヤー用JSONはプレーヤー情報だけを含む。動画名・解像度・総フレーム数は添付動画から取得する。
同じ対象のREQUESTを全クリップで共通に使い、動画が増えても本文は変わらない。
動画はファイル名を変更せず添付する。別のJSON SchemaやPROTOCOLファイルの添付は不要。
元動画の出典・ハッシュ・全PTS・フレーム対応はローカルのmanifestに保持する。
Pythonコードは配布せず、実装はgpt-6-astraに任せる。各REQUESTの対象別依頼文は
`resources/ball_detection/`または`resources/player_detection/`、JSON Schemaは
`runtime/contracts.py`から対象別に生成する。実装方法・ライブラリ・作業順序・応答の行数は指定しない。

`_preparation/`のmanifestと完成マーカーはローカルでの再実行検証用で、Chatには渡さない。
`project_kits/ball_detection/`と`project_kits/player_detection/`に、それぞれ2つのテキストを生成する。
各REQUESTのJSONは対象クラスの配列だけを含める。
複数動画・複数URLの準備でも、公開済み動画の整合性はローカルmetadataで検証する。
要求の版が異なる既存動画と混在する場合は明示的に失敗するため、新しいoutput_directoryを使う。
MCP提出版はキット5.0.0であり、4.xの準備済みmanifestを同じrootで再生成しない。旧動画を
新REQUESTで注釈する場合もmanifestは保持し、返却JSONはその元manifestで照合する。
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
- kit/prompt: 注釈型・補間上限・対象別要求文書で版を識別し、全クリップ共通のREQUESTを生成。
- runtime: repo内で使用する型、動画I/O、補完、検証、描画、ZIP処理。Chatには配布しない。
- resources: Project指示・成果物要求の唯一の保守元。

```bash
.venv/bin/python -m pytest tests/unit/tennis_scene/chat_annotation tests/e2e/tennis_scene/test_chat_annotation.py
.venv/bin/python -m ruff check src/tennis_scene/chat_annotation tests/unit/tennis_scene/chat_annotation tests/e2e/tennis_scene/test_chat_annotation.py
.venv/bin/python -m mypy src/tennis_scene/chat_annotation
```

YouTube取得はテストではmockにし、動画分割と配布キットは実際にエンコード・デコードする。
E2Eは動画とREQUEST本文の例から合成注釈を作り、ローカルmanifestを使って返却ZIPを検証する。
CFR/VFRの全フレーム・表示時間、参考区間の描画、JSON/overlay/ZIPの3成果物、ZIP内2ファイル構成、無効入力の拒否を確認する。
GPTが独自に作るコードやChat上での注釈結果そのものは自動テストの対象外。

ローカルで返却注釈を検証する場合は、元のmanifestを指定する（すべて絶対パス）。

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.annotate validate \
  --manifest /absolute/path/clip_manifest.json \
  --annotations /absolute/path/annotation_CLIP.json \
  --report /absolute/path/report.json
```

注釈の初期JSONは対象を指定して作成する。

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.annotate init \
  --manifest /absolute/path/clip_manifest.json \
  --video /absolute/path/clip.mp4 \
  --output /absolute/path/annotation_CLIP.json \
  --target ball # or player
```

同CLIの`finalize`は単一動画用で、`--video`、`--manifest`、`--annotations`、`--output`を受け取り、
有効なcompleted/partialだけをローカル確認用の注釈JSON、overlay動画、ZIPとして生成する。
このローカル確認用ZIPには前2点を格納するため、そのままMCPへ提出できない。MCPにはJSONのみをまとめ直して提出する。構造・入力エラーや
動画生成失敗ではZIPを公開せず非0終了する。既存の出力ディレクトリは上書きしない。


## MCP受領と注釈の整理

最小構成は **受領用MCP + ホスト上のtunnel-client + ローカルAI + 完了移動CLI**。
DB・ジョブキューは設けず、原本ZIP、採用済みJSON、AIの処理記録をファイルで保持する。

```text
ChatGPT → OpenAI Secure MCP Tunnel ← outbound HTTPS ← tunnel-client（remote PC）
                                                        ↓ localhost:8000/mcp
                                                   Docker: artifact-mcp-server
                                                        ↓ /artifacts bind mount
outputs/chat_annotation/
  annotated/
    raw/<sha256>.zip                        # JSONのみの提出原本、不変
    processing/<sha256>.json                # AIの判断・出典・保留・再開記録
    processed/ball/<clip_id>.json           # AIが検証・採用したボール注釈
    processed/player/<clip_id>.json         # AIが検証・採用した選手注釈
  videos/<source-video-name>/<clip_id>.mp4
  done/<source-video-name>/<clip_id>.mp4     # 両対象の完了後、機械的に移動
```

`artifacts/server.py`はStreamable HTTPの`/mcp`を提供する。公開ツールは次の3つ。

| ツール | 入力と結果 |
| --- | --- |
| `save_artifact(file, filename)` | ChatGPTのファイル参照からZIPを取得し、rawへ保存。artifact_id・SHA-256・サイズ・member一覧を返す |
| `list_artifacts(offset=0, limit=50)` | 受領ZIPの一覧。最大100件ずつ取得 |
| `read_artifact(artifact_id)` | ZIPのハッシュと内容を再検査し、member名とサイズを返す。JSON本文の取得・整理はローカルAIが担当 |

ChatGPTの[ファイル引数](https://developers.openai.com/plugins/reference#file-apis)を使い、
`file`を`openai/fileParams`へ宣言する。ChatGPTから渡されるdownload_urlのZIPをサーバーが取得する。
`sandbox:/...`やChatの表示リンクはリモートPCのファイルパスではない。
ZIP本体はTunnelのJSON引数に埋め込まず、許可したファイル配信ホストから別途HTTPSで取得する。
ファイル参照を渡せないChatGPT環境では未提出となるため、実環境で小さいZIPによる受入確認が必要。

ZIPは圧縮後16 MiB以下、展開後合計64 MiB以下、直下のJSON 1〜256件。
パス付きmember・symlink・暗号化・同名member・不正JSON・重複キーを拒否する。
スキーマや注釈の意味の判断は受領後のAIと既存validator関数が担当する。
保存名はZIPのSHA-256とし、同一バイトの再送を冪等に扱い、別内容は別原本として保存する。
処理途中のファイルを一覧に出さず、既存原本を上書きしない。

### 起動とTunnel接続

リモートPCのrepo/worktree rootで実行する。DockerはCPU専用の小さな依存環境で動く。
`CHAT_ANNOTATION_ROOT`は既存の準備済み動画があるrootの絶対パスにする。
CLI入口は`scripts/serve_artifacts.py`と`scripts/sync_done.py`。`--root`は必須の絶対パスで、
共通のパス契約で検証してから保存・移動処理へ渡す。

```bash
export CHAT_ANNOTATION_ROOT=/home/kamimura/projects/tennis-lab/outputs/chat_annotation
mkdir -p "$CHAT_ANNOTATION_ROOT/annotated/raw"
export ARTIFACT_UID="$(id -u)"
export ARTIFACT_GID="$(id -g)"
# 実際に使用するChatGPTファイル配信ホストを確認し、完全一致のホスト名を設定する。
# 以下は例。署名付きURL全体やワイルドカードは設定しない。
export ARTIFACT_DOWNLOAD_HOSTS=files.oaiusercontent.com,oaisdmntprcentralus.blob.core.windows.net,oaisdmntprjapaneast.blob.core.windows.net,oaisdmntprwestus3.blob.core.windows.net,oaisdmntprkoreacentral.blob.core.windows.net
docker compose -f src/tennis_scene/chat_annotation/artifacts/compose.yaml up -d --build
```

ホストの127.0.0.1:8000だけに公開し、rawのみをコンテナにマウントする。
認証はTunnelの組織・workspaceアクセスとruntime API keyを使用する。ローカルMCPには
独自Bearer認証を追加していないため、ポートの公開範囲を広げない。
許可ホスト以外・private IP・HTTP・redirectを拒否し、署名付きURLを保存しない。
取得URLの診断はサーバーログの`file download URL`行で行う。記録するのは
`scheme`・`hostname`・`port`だけで、URLのパス・署名クエリ・認証情報は含めない。
URL条件による拒否時はMCPエラーにもこの3項目と拒否条件名を返す。
`rejected=allowed_host`ならホスト未登録、`scheme`ならHTTPS以外、`port`なら非標準ポート、
`userinfo`/`fragment`なら禁止された認証情報/フラグメントを含むURLを意味する。
ホスト不一致の場合は実際の配信元を確認して許可リストと照合する。`sandbox`の場合は
ChatGPTから実際のファイル参照が渡っていないため、許可ホストを追加しても解決しない。
上記のAzure Blob 4ホストはChatGPTからの実提出で観測した配信先。環境・リージョンにより
配信先が異なる場合も、確認できた完全一致ホストだけを追加し、`*.blob.core.windows.net`のような
共有ドメインの一括許可は行わない。設定変更後は同じComposeコマンドでコンテナを再作成する。

Tunnelの作成・ChatGPT workspaceへの関連付け・権限は
[公式Secure MCP Tunnel手順](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)に従う。
`tunnel-client`は[公式release](https://github.com/openai/tunnel-client/releases/latest)から取得し、
リモートPC上で次を実行する。runtime用API keyは秘密管理から環境変数へ設定する。

```bash
# CONTROL_PLANE_API_KEYを環境に設定済みであること。
tunnel-client init --sample sample_mcp_remote_no_auth --profile chat-annotation \
  --tunnel-id tunnel_REPLACE_ME \
  --mcp-server-url http://127.0.0.1:8000/mcp
tunnel-client doctor --profile chat-annotation --explain
tunnel-client run --profile chat-annotation
```

ChatGPTのdeveloper-mode appで接続方式Tunnelと対象tunnelを選ぶ。
ツール一覧にsave_artifactが現れ、小さいJSON ZIPを送ってrawのSHA-256が応答と一致することを確認する。
このrepoへの実装だけではTunnelの作成・認証設定・ChatGPT接続は行われない。

Dockerを使わずに検証する場合は、通常のrepo環境を同期する。MCPの依存版はpyproject.tomlを正本とし、DockerもそこからMCPと共通パス検証に必要なOmegaConfだけをインストールする。

```bash
uv sync --locked
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.serve_artifacts \
  --root "$CHAT_ANNOTATION_ROOT/annotated/raw"
```

### AI処理と完了判定

raw→processedのAI作業の正本は[PROCESS_RAW.md](resources/PROCESS_RAW.md)。
AIの新しいセッションにはこのファイルとoutput rootを渡す。判断結果・出典ZIP/member・
出力ハッシュ・保留理由を`annotated/processing/`へ残すため、会話履歴に依存せず再開できる。
MCP受領後のAI起動はこの最小構成には含まれない。

完了判定は`artifacts/completion.py`で行う。対象ごとに正しいスキーマで、clip_id・解像度・
全フレーム・座標・補間が元manifestに適合し、ball/playerともcompletedの場合だけ移動する。
partial・片方未着はpending、不正JSONや競合はerrorsとして表示する。
準備receiptと動画SHA-256も検証する。doneの同名別ファイルを上書きせず、元のソース別階層を保持する。
videos/doneは同じファイルシステムに置く。移動はlink→unlinkで行い、中断後の再実行で再開する。

```bash
# AIの整理後に1回実行。dry-runはready一覧だけを表示する。
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.sync_done \
  --root "$CHAT_ANNOTATION_ROOT" --dry-run
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.sync_done \
  --root "$CHAT_ANNOTATION_ROOT"

# 常駐させる場合: 5秒ごとに両対象がそろったクリップから移動する。
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.sync_done \
  --root "$CHAT_ANNOTATION_ROOT" --watch-seconds 5
```

同じrootの複数completion実行はファイルロックで直列化する。単発実行でerrorsがあれば非0終了。
watcherはerrorsを標準出力へ報告して次周期も検査する。manifest自体の破損は処理を停止する。
processedはAIが一時ファイルからatomic renameで公開し、done移動済みの注釈の差替えは別途判断する。
準備処理の再実行はvideos/done両方の現所在とハッシュを検査する（同一キット版の範囲）。
