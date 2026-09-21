# ChatGPT Project用のテニス動画アノテーション準備

YouTube URLから動画を取得し、前後の参考区間を含む最大15秒・500,000,000 bytes以下の
MP4、元動画とのフレーム対応manifest、各Chatへの開始文、再利用するProjectキットを作る。
アノテーションの正本・座標・役割・補間・可視化・返却形式は
[PROTOCOL.md](resources/PROTOCOL.md)、機械契約は
[runtime/contracts.py](runtime/contracts.py)を参照する。JSON Schemaはこの型から生成する。

## 実行

repo/worktreeのrootから:

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.prepare \
  'source.url="https://www.youtube.com/watch?v=VIDEO_ID"'

# 長さは参考区間込み。20秒中、通常18秒を担当し前後各1秒を参考にする。
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.prepare \
  'source.url="https://www.youtube.com/watch?v=VIDEO_ID"' \
  clip.duration_seconds=20 clip.context_seconds=1

# 保存済み動画でオフライン確認。パスはpaths.data_rootからの相対パス。
.venv/bin/python -m src.tennis_scene.chat_annotation.scripts.prepare \
  source.local_video=samples/tennis_clip.mp4
```

全設定は[configs/prepare.yaml](configs/prepare.yaml)。CLI overrideも同じ設定を使用する。
URLとlocal_videoは排他的。HTTPSの単一YouTube動画URLを受け付け、playlist取得は行わない。
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

## 出力とProjectへの登録

```text
outputs/chat_annotation/
  sources/                         # 保存した元動画、YouTube取得metadata、ハッシュ記録
  project_kits/<kit-hash>/          # 初回登録する共通6ファイル
  videos/<source-id>/<run-hash>/
    prepared.json                  # 完成した全クリップの一覧、キットの場所
    clips/<clip-id>/
      <source-id>__<clip-id>.mp4
      clip_manifest.json
      REQUEST.txt
      ready.json                   # 検証済み入力一式の完成マーカー
```

1. ChatGPT Webで注釈用Projectを作成し、`project_kits/<kit-hash>`の**6ファイル全て**を
   Project Sourcesへ登録する。PROJECT_INSTRUCTIONS.txtはProject instructionsにも貼る。
2. 指定モデルを選び、Project内でクリップごとの新しいChatを作成する。
3. 該当クリップのMP4とclip_manifest.jsonを添付し、REQUEST.txtを貼る。共通キットの再添付は不要。
4. 初回はキット実ファイルの読込・preflight・画像表示・ZIPダウンロードまで確認する。
   Projectの共有と、利用アカウントでのコード実行・動画処理能力は別の条件。
5. 返却ZIPのvalidation_report.jsonと重畳動画を確認する。機械検証は意味的精度を保証しない。

返却ZIPには入力manifest原本、注釈、キット版、元URL/動画ID/ハッシュ、検証結果、
重畳MP4、一覧JPGが入る。ローカルでも配布されたannotation_tools.pyの同じコマンドで再現できる。
通常ChatでMP4展開やPython実行ができなければ、コードを推測して続けず制限を報告する。
Web UIの説明と初回手順は配布PROTOCOL.md末尾に集約している。

## 分割・再実行

表示順frame indexと元PTS/time_baseを保存し、CFR化・間引き・縮小を行わない。
フレーム境界に丸めるため長さは指定値以下になり、末尾は短くなる。
担当範囲は半開区間で全元フレームを一度ずつ覆う。前後の参考区間だけが重複する。
容量を超えたら担当範囲を二分して再エンコード。1フレーム＋文脈でも上限を超える場合は
明示的に失敗する。`-fs`による打ち切りや品質変更で成功扱いしない。

キット、元動画、設定のハッシュごとに出力を分離する。正常な完成物は再実行で検証して再利用し、
一時出力は完成物として扱わない。変更・破損した完成物はエラーにするため、新しいoutput_directoryを
指定するか、対象を確認して除去してから再実行する。自動で人の結果を上書きしない。
元動画は保持するため、取得と変換の両方に必要なディスク容量を確保する。

## 実装とテスト

- configuration/prepare/preparation: 厳密な設定、既存YouTube取得API、容量検査と公開。
- kit: repoのCourtKP20定義とportable runtimeから、版・ハッシュ付きの配布ファイルを生成。
- runtime: 型、動画I/O、補完、検証、描画、ZIP処理。配布時に最小限の共有パス検証も同梱し、インストール済みrepo・torch・モデル重みに依存しない。
- resources: Project指示・詳細プロトコル・短い開始文の唯一の保守元。

```bash
.venv/bin/python -m pytest tests/unit/tennis_scene/chat_annotation tests/e2e/tennis_scene/test_chat_annotation.py
.venv/bin/python -m ruff check src/tennis_scene/chat_annotation tests/unit/tennis_scene/chat_annotation tests/e2e/tennis_scene/test_chat_annotation.py
.venv/bin/python -m mypy src/tennis_scene/chat_annotation
```

YouTube取得はテストではmockにし、動画分割と配布キットは実際にエンコード・デコードする。
E2Eはrepo/GPU/モデル依存のimportを禁止した別プロセスで、返却ZIPまで実行する。
