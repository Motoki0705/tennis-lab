# Clip Studio

長時間・非同期のマルチカメラ動画をブラウザで同期し、ラリークリップを作成するローカルツールです。全カメラの fps・フレーム数・解像度が一致する形式で、統合パイプライン向けデータセットへ書き出します。

## 起動・再開

```bash
.venv/bin/python -m src.tennis_scene.scripts.clip_studio \
  source_directory=tennis_multivew/raw/meiji_3cam/video_000 gui.port=8765
```

画面は既定で `http://127.0.0.1:8765`、ポートは `gui.port` で指定します。

入力はDATA root相対の`tennis_multivew/raw/<dataset_id>/video_<3桁以上>/cam<index>.mp4`。
cameraはcam0から連番で、指定videoだけを開きます。保存先は
`tennis_multivew/processed/<dataset_id>/projects.json`（v1）、出力先はその隣の
`dataset/videos/<video_id>/clips/<clip_name>/`です。dataset/clip v2のスキーマとreaderは
[generate_dataset/manifest.py](../generate_dataset/manifest.py)が所有します。
旧raw直下の`cam0.mp4`や旧`dataset/clips/...`配置を自動探索・移行しません。

新規プロジェクト作成時だけ、全動画のコンテナの `creation_time` をUTCへそろえ、最も遅い録画開始を共通時刻0秒として同期オフセットを初期設定します。1つでも時刻が欠落・不正（タイムゾーンなしを含む）、または取得できない場合は、全カメラ0秒の従来動作で開始します。初期推定の結果と失敗理由は起動ログと画面上部に表示します。通知はそのサーバー起動中に表示され、通常の編集操作では消えません。既存JSONは、全オフセットが0秒・クリップ未登録でも再推定せず、そのまま読み込みます。初期値は映像を確認して音声同期・手動調整で修正できます。

## 保存済みデータのレビュー

```bash
# 既存projectだけを読み、編集・同期計算・書き出しをAPIでも拒否する。
.venv/bin/python -m src.tennis_scene.clip_studio \
  --data-root /absolute/path/to/data \
  --source-directory tennis_multivew/raw/meiji_3cam/video_000 --port 8904
```

`projects.json`・指定videoが無ければ停止します。rawと保存camera/pathの一致も必要です。
新projectの作成、同期値の再推定、dataset登録の修復、推論は行いません。

上部のraw → 同期project → 切出datasetで、現在のvideoの**保存clip数**と
**dataset.json登録数**・**未出力数**を区別します。保存clipが学習への採用や切出済みを
意味するわけではありません。この段階のデータはRGBと時間選別で、教師・学習splitはありません。
offsetの推定方法は既存project JSONからは分かりません。

停止すると、共通時刻から元動画時刻・0始まりの最近傍frameへの対応を全cameraで表示します。
プレビューと同じ`timeline.source_frame_index`を使い、録画範囲外はframeを`—`、映像なしと表示します。
再生中は厳密なframe照合を表示せず、停止を案内します。
一覧選択は再生位置を変えるだけです。選択clipの`[start,end)`、現在位置の区間内/外、
出力形式、元frame範囲（末尾を含む）・letterboxを確認できます。

| 表示 | 確認できた状態 |
|---|---|
| 登録済み | v2 clip・保存区間/同期値・dataset登録情報・mediaファイル存在が一致 |
| 未出力 | 保存区間に対応するclip.jsonがない |
| 出力不足 | clip.jsonまたはcamera mediaが欠ける |
| 不一致 | 出力manifestと保存project、またはdataset登録情報が異なる |
| 未登録 / 登録未確認 | clip/mediaは存在するが登録がない / indexがない・読めない |
| 確認失敗 | 旧version・不正manifest・現在のrawでは成立しない出力計画など、理由を表示 |

出力のfps・解像度は保存manifestを使って照合し、現在のexport既定設定とは区別します。
レビューは全動画の再decode、media内容の同一性、教師品質を検証しません。
同期・切出後の3D推定や教師採用の確認は各下流reviewの担当です。

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

## 書き出し設定

fps・解像度が異なる場合は `export.fps`、`export.width`、`export.height` を明示します。
解像度変換はアスペクト比を保つletterboxです。編集内容と既存clipの整合性を検証し、
再出力には `export.overwrite=true` を要求します。完成後のFPS・frame数・解像度も検証します。

## モジュール

- `web/service.py`：編集トランザクション、revision検証、自動保存、Undo/Redo。
- `__main__.py`：既存project専用のread-only CLI。`review.py`：保存projectの読込API。`web/review.py`：時刻対応と既存出力の照合。
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

ブラウザ回帰テストは `tests/e2e/tennis_scene/clip_studio_browser.mjs`。Playwrightのインストール先を `PLAYWRIGHT_MODULE`、60秒以上・2カメラ以上の**空の検証専用dataset `clip_studio_edit_test`**のURLを `CLIP_STUDIO_TEST_URL` に指定してNodeで実行します。テストはクリップと同期オフセットを変更します。実dataset名では編集前に拒否します。必要なら `CHROMIUM_PATH` でブラウザ実行ファイルを指定します。

再生要求の競合回帰テストは `node --test tests/e2e/tennis_scene/playback.test.mjs` で実行できます（Playwright不要）。

読取専用ブラウザー回帰は`tests/e2e/tennis_scene/clip_studio_review_browser.mjs`。
10fps・2camera・offset `[0,-1]`・`clip_000=[2,4)`・未出力の検証専用dataset
`clip_studio_review_test`を読取専用で開き、`CLIP_STUDIO_REVIEW_TEST_URL`と
`PLAYWRIGHT_MODULE`を指定します。実datasetでは起動前に拒否します。
frame照合・範囲外・半開区間・登録状態・API書込み拒否・再読込・狭い画面を検証します。
