# Clip Studio

長時間・非同期のマルチカメラ動画をブラウザで同期し、ラリークリップを作成するローカルツールです。全カメラの fps・フレーム数・解像度が一致する形式で、統合パイプライン向けデータセットへ書き出します。

## 編集モードで起動・再開する

リポジトリのルートで実行します。Python 環境（`.venv`）の準備は[インストールガイドの依存関係の手順](../../../docs/installation/README.ja.md#clone-and-sync)を参照してください。Clip Studio の同期・再生・書き出しは CPU で動作し、モデルの重みや CUDA 拡張は不要です。

1. DATA root（既定はリポジトリ直下の `data/`）の `tennis_multiview/raw/<dataset_id>/<video_id>/` に元動画を置きます。`video_id` は `video_000` のように `video_` と3桁以上の数字で構成し、動画は `cam0.mp4` から欠番なく並べます。対象フォルダ内の `.mp4` はすべてカメラ動画として扱うので、この命名にそろえてください。ブラウザで再生できる固定フレームレートの MP4（例 H.264）を使います。
2. 対象フォルダを DATA root 相対の `source_directory` に渡します。次は `meiji_4cam` の `video_000` を開く例です。別の `video_id` を開くときは、この引数を変更して起動し直します。

   ```bash
   .venv/bin/python -m src.tennis_scene.scripts.clip_studio \
     source_directory=tennis_multiview/raw/meiji_4cam/video_000
   ```

3. 起動ログが出たら、ブラウザで `http://127.0.0.1:8765` を開きます。ポートが使用中なら、起動コマンドに `gui.port=8766` を足して `http://127.0.0.1:8766` を開きます。終了は起動した端末で `Ctrl+C` です。

別の場所に `tennis_multiview/` がある場合は、その親ディレクトリを `paths.data_root` に指定します。

```bash
.venv/bin/python -m src.tennis_scene.scripts.clip_studio \
  paths.data_root=/absolute/path/to/data \
  source_directory=tennis_multiview/raw/meiji_4cam/video_000 gui.port=8766
```

初回は指定 `video_id` のプロジェクトを作成して `projects.json` に保存します。再開するときは同じ DATA root と `source_directory` を指定します。保存済みのカメラ ID とパスが現在の元動画と一致すれば、同期オフセットとクリップが復元されます。

新規作成時は全動画のコンテナの `creation_time` を UTC へそろえ、最も遅い録画開始が共通時刻 0 秒になるよう同期オフセットを初期化します。1つでも時刻が欠落・不正（タイムゾーンなしを含む）、または取得できない場合は、全カメラ 0 秒で開始します。結果と失敗理由は起動ログに出力し、サーバー起動中は画面上部にも表示します。初期値は映像を見ながら音声同期や手動調整で修正できます。再開時は保存済みの全オフセットが 0 秒でもその値を使います。

## 保存先とファイル構造

`meiji_4cam/video_000` で `clip_000` を書き出した場合の配置です。元動画のディレクトリから dataset ID と video ID を読み取り、同じ DATA root の `processed/` 以下に保存します。

```text
<DATA root>/
└── tennis_multiview/
    ├── raw/
    │   └── meiji_4cam/
    │       └── video_000/
    │           ├── cam0.mp4
    │           ├── cam1.mp4
    │           ├── cam2.mp4
    │           └── cam3.mp4
    └── processed/
        └── meiji_4cam/
            ├── projects.json
            └── dataset/
                ├── dataset.json
                └── videos/
                    └── video_000/
                        └── clips/
                            └── clip_000/
                                ├── clip.json
                                └── media/
                                    ├── cam0.mp4
                                    ├── cam1.mp4
                                    ├── cam2.mp4
                                    └── cam3.mp4
```

| ファイル | 保存される内容 | 保存のタイミング |
|---|---|---|
| `projects.json` | データセット内の動画ごとの元動画パス、同期オフセット、クリップ名・区間、試合設定・カウント | 初回起動で対象動画を登録。以後は編集操作ごとに自動保存 |
| `dataset/dataset.json` | 書き出し済みクリップの一覧。各項目に `clip_id`（例 `video_000/clip_000`）、`dataset/` からの相対パス、カメラ数、fps、フレーム数、解像度を保持 | 各クリップの書き出しが完了すると登録 |
| `…/clip_000/clip.json` | 書き出した区間・形式と、カメラ別の元動画パス、同期オフセット、元動画の fps、使用した最初と最後のフレーム番号（0始まり、末尾を含む）、余白の設定 | `media/` と一緒に公開 |
| `…/clip_000/media/cam*.mp4` | 全カメラで fps・フレーム数・解像度をそろえた音声なしの H.264 MP4 | 全カメラの出力検証後に公開 |

クリップを追加した段階で保存されるのは編集状態です。MP4 を作るには「このクリップを書き出す」か「全クリップを書き出す」を押します。区間の変更や削除は `projects.json` に反映され、すでに出力したファイルや `dataset.json` の登録は保持されます。区間や同期値を変更したクリップを同じ名前で更新する手順は[書き出し設定](#書き出し設定)を参照してください。試合設定・カウントは `projects.json` に保持し、必要に応じて[カウントCSV](#カウントcsv)をダウンロードします。

### `projects.json` の例（v2）

```json
{
  "version": 2,
  "dataset_id": "meiji_4cam",
  "projects": {
    "video_000": {
      "sources": [
        {"camera_id": "cam0", "path": "tennis_multiview/raw/meiji_4cam/video_000/cam0.mp4", "offset_sec": 0.0},
        {"camera_id": "cam1", "path": "tennis_multiview/raw/meiji_4cam/video_000/cam1.mp4", "offset_sec": 1.2},
        {"camera_id": "cam2", "path": "tennis_multiview/raw/meiji_4cam/video_000/cam2.mp4", "offset_sec": 0.3},
        {"camera_id": "cam3", "path": "tennis_multiview/raw/meiji_4cam/video_000/cam3.mp4", "offset_sec": 0.8}
      ],
      "clips": [
        {"name": "clip_000", "start_sec": 12.0, "end_sec": 17.0, "score": null}
      ],
      "match": null
    }
  }
}
```

各 `sources[].path` は DATA root 相対です。共通時刻に `offset_sec` を足すと各元動画の時刻になるため、例のクリップ `[12.0, 17.0)` は `cam1` の `[13.2, 18.2)` に対応します。終了時刻は区間に含みません。`match` と `score` の `null` は未入力です。v1 の編集データも開けます。試合設定とカウントは未入力で読み込み、次に保存した時点で v2 に更新します。

`clip.json` の `video_paths` と `cameras[].video` はクリップディレクトリ相対、`cameras[].source_path` は書き出し時の元動画の絶対パスです。`dataset.json` と `clip.json` の v2 スキーマは [generate_dataset/manifest.py](../generate_dataset/manifest.py) を参照してください。

編集モードの Hydra ログは `paths.output_root`（既定 `outputs/`）に出力します。編集状態や動画を探すときは DATA root の `tennis_multiview/processed/<dataset_id>/` を確認してください。

## 保存済みデータのレビュー

```bash
.venv/bin/python -m src.tennis_scene.clip_studio \
  --data-root "$PWD/data" \
  --source-directory tennis_multiview/raw/meiji_4cam/video_000 --port 8904
```

同じくリポジトリのルートから起動し、ブラウザで `http://127.0.0.1:8904` を開きます。DATA root を変更して編集した場合は `--data-root` にも同じ場所を指定してください。レビューは読み取り専用です。

`projects.json`・指定videoが無ければ停止します。rawと保存camera/pathの一致も必要です。
新projectの作成、同期値の再推定、dataset登録の修復、推論は行いません。

以下の照合表示（dataset関係・元frame対応）は、読取専用レビューとカメラ比較で表示します。
編集時の単一カメラ（クリップ作成）では表示せず、元frame対応の照合要求も止めます。

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

### 旧ディレクトリ名からの移行

旧配置の `raw/<dataset_id>/cam0.mp4` や `dataset/clips/` を使っていた場合は、起動前に動画 ID ごとの配置と v2 のデータセット構造を用意してください。Clip Studio が読み込む経路は[保存先とファイル構造](#保存先とファイル構造)のツリーのとおりです。

1. データセットを使用する編集・書き出しを停止し、移行前のメタデータをバックアップします。
2. 旧綴りのディレクトリを DATA root 直下の `tennis_multiview` に改名します。
   旧ディレクトリが DATA root 外を指すシンボリックリンクの場合は、実体を DATA root 内の
   `tennis_multiview` へコピーし、参照先まで DATA root に収まるようにします。
3. `projects.json` の `sources[].path`、`clip.json` の `cameras[].source_path` など、
   保存済み JSON のパスに含まれる旧ディレクトリ名を更新します。
   `sources[].path` は DATA root 相対のまま、絶対パスは移行先を指すようにします。
4. 上記の読取専用 CLI で対象 video を開き、元動画と保存済みクリップの登録状態を確認します。

## 編集の流れ

1. **同期を確認**：「同期調整を開く」でカメラ比較へ切り替え、全カメラと動画横の「カメラの同期調整」を表示する（狭い画面では動画直下に表示）。基準カメラを選んで音声同期を計算し、候補の時差と信頼度を確認して適用する。必要に応じて各カメラの時差を数値入力または ±1フレームで調整する。計算後に編集した場合は候補を再計算する。終わったら「同期終了」を押す。
2. **ラリーを探す**：「同期終了」または「単一カメラ」で、フォーカス中のカメラだけを大きく表示する。シークバーをクリック／ドラッグするか、時刻を入力して移動する。タイムラインは拡大・縮小・左右移動できる。「⛶ 全画面」または `F` で全画面にできる。
3. **切り出す**：開始 `I` → 終了 `O` → 追加 `C`。開始・終了マークはクリップ区間と同じセグメントバーに表示される。追加後はマークが解除され、同じ位置から次のラリーを探せる。
4. **修正する**：一覧からクリップを選び、名前・開始・終了を変更して保存する。「現在位置を開始／終了に」と区間リピートも利用できる。
5. **書き出す**：「このクリップを書き出す」または「全クリップを書き出す」を押す。進捗欄で完了を確認し、`dataset/` から出力を取り出す。

### クリップのカウント

1. 「試合設定とカウントCSV」で、動画ごとに固定の選手・ペア A/B の表示名、採点方式、最初のサーブ側を設定します。
2. 保存済みクリップを選び、「クリップのカウント」で開始前カウントを入力します。直前の終了後カウントを使う場合は「直前から連鎖」を選びます。サーブ側は自動または A/B の手動入力を選べます。
3. 「A の得点」または「B の得点」で確定すると終了後カウントが表示され、開始時刻順の次の未入力クリップに仮入力されます。カウント一覧と選択中の表示は読取専用レビューでも確認できます。

「不明」「複数ポイント」には理由を入力します。そこで連鎖が止まるので、次の開始前カウントは手入力します。「ポイント外」には理由を入力し、前の終了後カウントを次へ引き継ぎます。通過するカウントがない場合のサーブ側は手動で指定します。「カウントを消去」で未入力に戻ります。

先行クリップのカウントを訂正すると、次に手入力した開始前カウントか、連鎖が止まるクリップの手前まで再計算します。保存の確認画面には、再計算で変わる後続クリップ数と、新たに連鎖切れになるクリップの件数・名前が出ます。連鎖切れのクリップは保存済みの状態・得点側・理由を表示します。開始前カウントを手入力するか上流を修正して連鎖が戻ると、終了後の表示と後続への連鎖が再開します。

開始前の手入力が直前の終了後と食い違う場合は「不連続」と表示し、手入力の値から連鎖を再開します。同じカウントで直前から計算したサーブ側と保存値が食い違う場合は「サーブ順の矛盾」と表示します。硬式の第2セット以降では、開始前を手入力したクリップの保存済みサーブ側を、そのクリップ以降のサーブ順の基準とします。ソフトテニスと硬式の第1セットでは試合設定からサーブ順を計算します。連鎖内でサーブ側だけを手入力すると、そのクリップの表示・保存に反映します。

- **ソフトテニス**は奇数のゲーム数で設定し、通常は 4 点先取・2 点差でゲームを取ります。最終ゲームは 7 点先取・2 点差で、サーブ側を 2 点ごとに交代します。[日本ソフトテニス連盟のルール](https://www.jsta.or.jp/wp-content/uploads/2016/02/rule_match.pdf)
- **硬式テニス**は 3 または 5 セットマッチで、ノーアドと最終セット形式を設定できます。10 点マッチタイブレークはセット数が並んだ時点で最終セットに代えて行います。硬式の第 2 セット以降で、以前のゲーム数が分からない区間から記録を始めるときは、最初のクリップのサーブ側を手入力してサーブ順を確定します。[ITF Rules of Tennis 第 5・6・14 条、付録 VI](https://www.world.tennis/media/7221/2026-rules-of-tennis-english.pdf)

ポイントは A/B の順で整数を入力します。通常ゲームは 0〜3 点とアドバンテージの 4 点で表し、硬式では 0・15・30・40・AD に変換して表示します。ファイナルゲームやタイブレークのデュース以降は、サーブ順を保つため計 4 点単位で正規化します。7 点先取ならデュースは 6–6 または 7–7、10 点先取なら 9–9 または 10–10 です。

クリップの `score` には、開始前カウントを `before` の `games`、`points`、硬式の `sets` として格納します。ポイント外の記録は `before` を省きます。終了後カウントは確定済みの開始前と得点側から計算します。

### カウントCSV

「試合設定とカウントCSV」から UTF-8 CSV を取り込み、ダウンロードできます。ヘッダーは以下の順と列名を使用します。

```csv
clip_name,status,sets_a,sets_b,games_a,games_b,points_a,points_b,server,winner,note
```

- `clip_name` は保存済みクリップの名前です。`status` は `auto`（仮入力）、`confirmed`（確定）、`unknown`（不明）、`not_point`（ポイント外）、`multiple_points`（複数ポイント）です。
- `server` と確定時の `winner` は `A` または `B` を指定します。確定以外の `winner`、ソフトテニスのセット列、ポイント外の各カウント列は空欄です。不明・ポイント外・複数ポイントの `note` には理由を書きます。
- ダウンロードには入力済みクリップが含まれます。連鎖切れがあると該当クリップ名を表示し、CSV出力を止めます。取り込んだ行だけを開始前の手入力（`anchor`）・手動サーブ（`manual`）として保存します。存在しない名前、重複行、ルール違反を含め全行を検証し、全行が有効なら 1 回の操作として保存します。再ダウンロードで CSV 列の値を確認できます。

### カメラの読込

動画の読込はカメラ数に比例して重くなるため、表示中のカメラだけを読み込みます。

| 状態 | 読み込むカメラ |
|---|---|
| 単一カメラ（起動直後・「同期終了」後） | フォーカス中の1台。カメラを切り替えると前のカメラを解放する |
| カメラ比較・同期調整中 | 全カメラ。開くたびにその場で読み込む |

解放したカメラは、`<video>` のソースとバッファ、停止フレームの画像を破棄し、
取得中の `/api/frame/{camera}` 要求を中断します。以後そのカメラの停止フレームは取得しません。

### 全画面

単一カメラのときだけ、映像と再生操作を全画面にできます（「⛶ 全画面」・`F`、終了は同じ操作か `Esc`）。
全画面では、再生ボタン・時刻・シークバーと、開始/終了マークとクリップ区間のセグメントバーだけを下部に残します。
マーク・追加はキーボード（`I` / `O` / `C`）で行います。

### 再生・移動

| 操作 | 挙動 |
|---|---|
| Space / 再生ボタン | 再生・停止 |
| 再生速度（⚙） | 0.25 / 0.5 / 1 / 1.5 / 2 / 4倍 |
| ← / → | 移動量セレクタで指定した量だけ移動して停止 |
| 移動量（⚙） | 1フレーム / 0.1 / 1 / 5 / 10 / 30秒 |
| Shift + ← / → | 選択移動量の10倍 |
| `,` / `.` | 選択カメラのfpsで常に1フレーム移動 |
| A / B | 選択クリップに A / B の得点を記録して確定 |
| Ctrl/Cmd + Z / Shift + Z | 取り消し / やり直し |
| F | 単一カメラの全画面表示・終了 |

再生速度と移動量は、映像右上の歯車（⚙、全画面でも表示）から開くメニューで変更します。メニューにはキー操作の一覧もあります。再生速度が1倍以外のときは歯車に倍率を表示します。再生速度と矢印キー移動量は独立しています。数値・名前・選択肢を入力中は編集を優先し、ショートカットを抑制します。

各カメラ映像にポインターを置いて `Ctrl + ホイール`（macOSは `Cmd + ホイール`）を操作すると、その位置を固定したまま1～8倍へ拡大・縮小できます。倍率と注視位置はカメラごとに独立しており、停止フレームと再生動画で共通です。「等倍」でそのカメラだけ100%表示へ戻せます。

再生はブラウザの動画デコーダーを使用し、カメラ比較では選択カメラの動画時刻に他カメラを追従させます。停止・シーク時はPython側で取得した確認フレームと、そのフレーム番号を表示します。**再生中の複数動画は厳密な同時フレーム表示ではないため、同期の最終確認は停止状態で行ってください。** 全動画はミュート再生です。選択カメラに映像がない時刻からは再生を開始できません。

ブラウザで再生可能な形式（H.264 MP4など）が必要です。非対応形式は画面にエラーを表示し、自動変換はしません。停止フレーム取得はOpenCV側の対応範囲で利用できます。シーク応答は元動画の解像度・圧縮方式・キーフレーム間隔にも依存します。映像区間の定義は一定fpsを前提とします。

### 保存と長時間処理

クリップ作成・変更・削除、同期変更、試合設定とカウント編集は、その都度JSONへatomic writeで自動保存します。連鎖再計算とCSV取り込みを含め、取り消し履歴はサーバー起動中の最新100操作です。保存失敗時は操作を適用せず、エラーを表示します。複数タブの古い編集要求は拒否されるため、「再読込」してからやり直してください。再生位置や表示モードはプロジェクトJSONには保存しません。

音声同期と書き出しは1つのバックグラウンドワーカーで実行し、画面を操作しながら進捗を確認できます。書き出しは開始時の保存済み編集状態を使用します。**書き出しのキャンセルは実行中のエンコードプロセスを停止**し、そのクリップの途中ファイルを削除します。完成・公開済みクリップは保持します。動画は一時領域で全カメラの検証まで終えてから公開するため、中断したクリップをそのまま再実行できます。カメラ別のフレーム数で進捗を表示します。音声同期のキャンセルは解析結果を破棄します。サーバー停止後に実行中ジョブを再開する機能はありません。

## 書き出し設定

全カメラの fps と解像度が一致していれば、既定ではその値を使います。異なる場合や出力形式を変えたい場合は、起動時に次のように指定します。幅と高さは正の偶数をセットで渡します。解像度を変える場合、アスペクト比を保って余白を追加します（letterbox）。

```bash
.venv/bin/python -m src.tennis_scene.scripts.clip_studio \
  source_directory=tennis_multiview/raw/meiji_4cam/video_000 \
  export.fps=30 export.width=1920 export.height=1080
```

書き出し前に、対象クリップの各出力フレームに全カメラの元フレームが存在することを確認します。既定の `export.overwrite=false` では編集内容と設定に一致する完成済みクリップを再利用し、不一致や不完全な出力を検出するとエラーになります。同じクリップ名で置き換える場合は、`export.overwrite=true` を追加して起動し直し、対象クリップを書き出します。設定の既定値は [clip_studio.yaml](../configs/clip_studio.yaml) を参照してください。

## モジュール

- `web/service.py`：編集トランザクション、revision検証、自動保存、Undo/Redo。
- `__main__.py`：既存project専用のread-only CLI。`review.py`：保存projectの読込API。`web/review.py`：時刻対応と既存出力の照合。
- `web/app.py`：FastAPI、HTTP Range動画配信、停止時のJPEGフレーム取得。ループバックで起動する。
- `web/jobs.py`：同期候補計算・バッチ事前検証・出力済み判定・進捗。
- `web/exporting.py`：停止可能なエンコード子プロセス、一時出力、公開とロールバック。
- `web/static/`：ビルド不要のHTML/CSS/JavaScript。`playback.js` が再生・カメラ別の読込と解放・確認フレーム取得、`studio.js` が編集操作・表示モード・全画面・設定メニュー、`review.js` がdataset照合表示を担当する。
- `initialization.py`：新規プロジェクトの録画時刻による初期同期と、既存プロジェクトの復元。
- `layout.py`：raw dataset/videoからcamera、`projects.json`、dataset出力先を厳密に導出。
- `project.py`：dataset単位の`projects.json`とvideo projectの単一定義・I/O。
- `score_chain.py` / `score_csv.py`：クリップ間の再計算とCSV入出力。採点ルールは共通の [tennis_scoring.py](../../utils/tennis_scoring.py) を使用する。
- `timeline.py`：`local_time = global_time + offset_sec`、区間 `[start_sec, end_sec)` と最近傍フレームの写像。
- `state.py`：cv2非依存のタイムライン状態。
- `sources.py`：smart-seek・縮小・LRU付きのプレビュー取得。
- `audio_sync.py`：音声エンベロープのFFT相互相関。
- `export.py` / `imaging.py`：検証済み出力計画、動画書き出し、letterbox。
- `app.py` / `render.py`：従来のOpenCVウィンドウ実装。起動スクリプトはブラウザ版を使用する。

## 検証

```bash
.venv/bin/python -m pytest tests/unit/utils/test_tennis_scoring.py tests/unit/tennis_scene/clip_studio \
  tests/unit/tennis_scene/test_configuration.py \
  tests/integration/tennis_scene/test_clip_studio_web.py \
  tests/integration/tennis_scene/test_clip_export.py
```

ブラウザ回帰テストは `tests/e2e/tennis_scene/clip_studio_browser.mjs`。Playwrightのインストール先を `PLAYWRIGHT_MODULE`、60秒以上・2カメラ以上の**空の検証専用dataset `clip_studio_edit_test`**のURLを `CLIP_STUDIO_TEST_URL` に指定してNodeで実行します。テストはクリップ・同期オフセット・試合設定・カウントを変更します。実dataset名では編集前に拒否します。必要なら `CHROMIUM_PATH` でブラウザ実行ファイルを指定します。

再生要求の競合とカメラ読込制御の回帰テストは `node --test tests/e2e/tennis_scene/playback.test.mjs` で実行できます（Playwright不要）。

読取専用ブラウザー回帰は`tests/e2e/tennis_scene/clip_studio_review_browser.mjs`。
10fps・2camera・offset `[0,-1]`・`clip_000=[2,4)`・未出力の検証専用dataset
`clip_studio_review_test`を読取専用で開き、`CLIP_STUDIO_REVIEW_TEST_URL`と
`PLAYWRIGHT_MODULE`を指定します。実datasetでは起動前に拒否します。
frame照合・範囲外・半開区間・登録状態・カウント表示・API書込み拒否・再読込・狭い画面を検証します。
サーブ順の警告も検証する場合は、硬式・最初のサーブ A の試合で、`clip_000` を第1セット 5–1・3–0 から A 得点の確定、`next_set=[5,6)` を開始前手入力・第2セット先頭・保存サーブ A・B 得点の確定として用意します。
