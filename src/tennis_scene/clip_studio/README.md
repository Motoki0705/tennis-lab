# Clip Studio

長時間・非同期のマルチカメラ動画をブラウザで同期し、ラリークリップを作成するローカルツールです。全カメラの fps・フレーム数・解像度が一致する形式で、統合パイプライン向けデータセットへ書き出します。

## 起動・再開

リポジトリのルートで実行します。ブラウザで表示された `http://127.0.0.1:8765` を開いてください。別のポートは `gui.port=8766` で指定します。

```bash
# 新規作成・再開とも、対象videoディレクトリだけを指定する
.venv/bin/python -m src.tennis_scene.scripts.clip_studio \
  source_directory=tennis_multivew/raw/meiji_3cam/video_002
```

`source_directory`は`paths.data_root`（既定`data/`）からの相対パスで、必ず
`tennis_multivew/raw/<dataset_id>/<video_id>`形式とします。`video_id`は
`video_000`のように3桁以上の数字を使い、直下の`cam0.mp4`から欠番なく
cameraを検出します。別名のMP4、camera番号の欠番、標準外の階層は起動前に
エラーになります。

projectと出力先はsourceから一意に導出されます。

```text
data/tennis_multivew/
├── raw/<dataset_id>/<video_id>/cam<index>.mp4
└── processed/<dataset_id>/
    ├── projects.json
    └── dataset/
        ├── dataset.json
        └── videos/<video_id>/clips/<clip_name>/
            ├── clip.json
            └── media/<camera_id>.mp4
```

`projects.json`はdataset内のvideoごとにsource、同期offset、clip編集を保持します。
同じ`source_directory`を再指定すると該当videoを再開し、保存済みsourceと現在の
cameraファイルが異なる場合はエラーになります。専用worktreeから元repoのdataを
利用する場合は`paths.data_root=/元repoの絶対パス/data`を追加します。

旧`raw/<recording_id>`と`dataset/clips/<recording_id>/<clip_name>`構造は読み込みません。
既存データを一度だけ移行する場合は、先にdry-runし、同じ引数へ`--apply`を追加します。

```bash
.venv/bin/python -m src.tennis_scene.scripts.migrate_video_clip_layout \
  --data-root /absolute/path/to/tennis-lab/data \
  --dataset-id meiji_3cam
```

新規プロジェクト作成時だけ、全動画のコンテナの `creation_time` をUTCへそろえ、最も遅い録画開始を共通時刻0秒として同期オフセットを初期設定します。1つでも時刻が欠落・不正（タイムゾーンなしを含む）、または取得できない場合は、全カメラ0秒の従来動作で開始します。初期推定の結果と失敗理由は起動ログと画面上部に表示します。通知はそのサーバー起動中に表示され、通常の編集操作では消えません。既存JSONは、全オフセットが0秒・クリップ未登録でも再推定せず、そのまま読み込みます。初期値は映像を確認して音声同期・手動調整で修正できます。

## 編集の流れ

1. **同期を確認**：「カメラ比較」で全カメラを表示し、動画横の「カメラの同期調整」を開く（狭い画面では動画直下に表示）。基準カメラを選んで音声同期を計算し、候補の時差と信頼度を確認して適用する。必要に応じて各カメラの時差を数値入力または ±1フレームで調整する。同期後はパネルを閉じて動画領域を広く使える。計算後に編集した場合は候補を再計算する。
2. **ラリーを探す**：「単一カメラ」で選択カメラを大きく表示する。シークバーをクリック／ドラッグするか、時刻を入力して移動する。タイムラインは拡大・縮小・左右移動できる。
3. **切り出す**：開始 `I` → 終了 `O` → 追加 `C`。追加後はマークが解除され、同じ位置から次のラリーを探せる。
4. **修正する**：一覧からクリップを選び、名前・開始・終了を変更して保存する。「現在位置を開始／終了に」と区間リピートも利用できる。
5. **書き出す**：選択クリップまたは全クリップを出力する。fps・解像度の不一致や範囲外は、バッチ全体の事前検証で明示的にエラーにする。

### 再生・移動

| 操作 | 挙動 |
|---|---|
| Space / 再生ボタン | 再生・停止 |
| 再生速度 | 0.25 / 0.5 / 1 / 1.5 / 2 / 4倍 |
| ← / → | 移動量セレクタで指定した量だけ移動して停止 |
| 移動量 | 1フレーム / 0.1 / 1 / 5 / 10 / 30秒 |
| Shift + ← / → | 選択移動量の10倍 |
| `,` / `.` | 選択カメラのfpsで常に1フレーム移動 |
| Ctrl/Cmd + Z / Shift + Z | 取り消し / やり直し |

再生速度と矢印キー移動量は独立しています。数値・名前・選択肢を入力中は編集を優先し、ショートカットを抑制します。

各カメラ映像にポインターを置いて `Ctrl + ホイール`（macOSは `Cmd + ホイール`）を操作すると、その位置を固定したまま1～8倍へ拡大・縮小できます。倍率と注視位置はカメラごとに独立しており、停止フレームと再生動画で共通です。「等倍」でそのカメラだけ100%表示へ戻せます。

再生はブラウザの動画デコーダーを使用し、選択カメラの動画時刻に他カメラを追従させます。停止・シーク時はPython側で取得した確認フレームと、そのフレーム番号を表示します。**再生中の複数動画は厳密な同時フレーム表示ではないため、同期の最終確認は停止状態で行ってください。** 全動画はミュート再生です。選択カメラに映像がない時刻からは再生を開始できません。

ブラウザで再生可能な形式（H.264 MP4など）が必要です。非対応形式は画面にエラーを表示し、自動変換はしません。停止フレーム取得はOpenCV側の対応範囲で利用できます。シーク応答は元動画の解像度・圧縮方式・キーフレーム間隔にも依存します。映像区間の定義は一定fpsを前提とします。

### 保存と長時間処理

クリップ作成・変更・削除と同期変更は、その都度JSONへatomic writeで自動保存します。保存失敗時は操作を適用せず、エラーを表示します。取り消し履歴はサーバー起動中の最新100操作です。複数タブの古い編集要求は拒否されるため、「再読込」してからやり直してください。再生位置や表示モードはプロジェクトJSONには保存しません。

音声同期と書き出しは1つのバックグラウンドワーカーで実行し、画面を操作しながら進捗を確認できます。書き出しは開始時の保存済み編集状態を使用します。**書き出しのキャンセルは実行中のエンコードプロセスを停止**し、そのクリップの途中ファイルを削除します。完成・公開済みクリップは保持します。動画は一時領域で全カメラの検証まで終えてから公開するため、中断したクリップをそのまま再実行できます。カメラ別のフレーム数で進捗を表示します。音声同期のキャンセルは解析結果を破棄します。サーバー停止後に実行中ジョブを再開する機能はありません。

## エクスポート形式

`<dataset_root>/dataset.json`にインデックスを持ち、各クリップは
`videos/<video_id>/clips/<clip_name>/clip.json`と`media/<camera_id>.mp4`を
持ちます。clip IDは`<video_id>/<clip_name>`なので、別videoで同じclip名を
使用できます。マニフェストの`video_paths` / `camera_ids`は統合パイプラインの
入力として利用できます。

fps・解像度がカメラ間で異なる場合は、起動時に `export.fps=30 export.width=1280 export.height=720` のように明示指定します。解像度変換はアスペクト比を保つletterboxです。既存出力は既定では上書きしません。ブラウザの一括・個別書き出しでは、現在の区間・同期・ソース・出力仕様とマニフェストが一致し、全動画のfps・フレーム数・解像度も検証できたものだけを「出力済み」としてスキップします。同名でも編集内容が変わっている場合や、不完全な出力は明示的にエラーにします。再出力は `export.overwrite=true` で明示します。書き出し後はfps・フレーム数・解像度を再検証します。疑似アノテーションの追加方法は [`../generate_dataset/README.md`](../generate_dataset/README.md) を参照してください。

```bash
# 保存済みプロジェクトからヘッドレス出力
.venv/bin/python -m src.tennis_scene.scripts.export_clips \
  source_directory=tennis_multivew/raw/meiji_3cam/video_002
```

## モジュール

- `web/service.py`：編集トランザクション、revision検証、自動保存、Undo/Redo。
- `web/app.py`：FastAPI、HTTP Range動画配信、停止時のJPEGフレーム取得。ループバックで起動する。
- `web/jobs.py`：同期候補計算・バッチ事前検証・出力済み判定・進捗。
- `web/exporting.py`：停止可能なエンコード子プロセス、一時出力、公開とロールバック。
- `web/static/`：ビルド不要のHTML/CSS/JavaScript。`playback.js` が再生と確認フレーム取得、`studio.js` が編集操作と画面更新を担当する。
- `initialization.py`：新規プロジェクトの録画時刻による初期同期と、既存プロジェクトの復元。
- `layout.py`：raw dataset/videoからcamera、`projects.json`、dataset出力先を厳密に導出。
- `project.py`：dataset単位の`projects.json`とvideo projectの単一定義・I/O。
- `timeline.py`：`local_time = global_time + offset_sec`、区間 `[start_sec, end_sec)` と最近傍フレームの写像。
- `state.py`：cv2非依存のタイムライン状態。
- `sources.py`：smart-seek・縮小・LRU付きのプレビュー取得。
- `audio_sync.py`：音声エンベロープのFFT相互相関。
- `export.py` / `imaging.py`：検証済み出力計画、動画書き出し、letterbox。
- `app.py` / `render.py`：従来のOpenCVウィンドウ実装。起動スクリプトはブラウザ版を使用する。

## 検証

```bash
.venv/bin/python -m pytest tests/unit/tennis_scene/clip_studio \
  tests/unit/tennis_scene/test_configuration.py \
  tests/integration/tennis_scene/test_clip_studio_web.py \
  tests/integration/tennis_scene/test_clip_export.py
```

ブラウザ回帰テストは `tests/e2e/tennis_scene/clip_studio_browser.mjs`。Playwrightのインストール先を `PLAYWRIGHT_MODULE`、60秒以上・2カメラ以上の**空の検証専用プロジェクト**のURLを `CLIP_STUDIO_TEST_URL` に指定してNodeで実行します。テストはクリップと同期オフセットを変更します。必要なら `CHROMIUM_PATH` でブラウザ実行ファイルを指定します。

再生要求の競合回帰テストは `node --test tests/e2e/tennis_scene/playback.test.mjs` で実行できます（Playwright不要）。
