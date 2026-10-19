# Shared Visualization

## タスク別の使い方

起動コマンドと操作手順は各ガイドを正本とします。

| タスク | ガイド | 閲覧 / 推論ポート |
|---|---|---|
| PLCS | [利用ガイド](../../plcs/visualization/README.md) | 8772 / 8771 |
| BLCS | [利用ガイド](../../blcs/visualization/README.md) | 8773 / 8770 |
| Court Detection | [利用ガイド](../../court_detection/visualization/README.md) | 8774 / 8775 |
| Ball Detection | [利用ガイド](../../ball_detection/visualization/README.md) | 8776 / 8777 |
| Ball Refiner (座標2D/3D) | [利用ガイド](../../ball_refiner/coordinates/review/README.md) | 8784（閲覧・拡張・推論を統合） |

## 実行前確認

- コードのあるリポジトリまたはworktreeの直下で実行します。例えば現在のPR作業用なら`cd /home/kamimura/projects/tennis-lab/.claude/worktrees/dataset-scene-review`です。マージ後は通常のリポジトリ直下で使えます。
- `scripts/run_in_repo_venv.sh`は元リポジトリの`.venv/bin/python`を使います。仮想環境がない場合は、元リポジトリで`uv sync --locked`を実行してください。
- ガイド中の`ROOT`はGitの共通ディレクトリから元リポジトリを解決し、既存のdata・outputs・ckptを参照します。コードは現在の作業ディレクトリのものを実行します。
- データと重みは別途配置が必要です。UIはダウンロード・データ生成を行いません。閲覧だけならGPUは不要です。
- 信頼できるcheckpointだけを使ってください。PyTorchのcheckpointはpickleを含み、読み込み時にコードを実行し得ます。
- ローカル利用向けです。フロントエンドのビルドは不要で、URLはサーバーを起動した端末から開きます。

## 構成

- [review](review/README.md): BLCS・PLCSの生成シーン閲覧API・データ契約。
- [detection](detection/README.md): Court・Ball Detectionの画像閲覧・推論Web UI。
- `shared/scene3d.mjs`: Three.jsによるコート・カメラ・GTと予測の描画。
- `web_assets.py`: 固定されたローカル描画アセットのHTTP配信。
- `inference_queue.py`: 推論Web UIから共有GPUキューへの接続。
- `../scripts/inference_worker.py`: キューが起動するCLI。共有要求ディレクトリ内のパスを検証してから推論を実行する。

## 発表用ビジュアル（presentation）

`presentation/` は matplotlib の呼び出し元が所有する Axes に描画し、Agg canvas の
RGB `uint8` 配列を MP4・GIF・静止画へ書き出す共通部品です。task 固有の入力読込・
場面構成・Hydra CLI は各 task の `visualization/presentation/` に置きます。

- `Segment` / `StepClock`: 累積秒数を最近傍の frame 境界へ丸めます（中間値は切り上げ）。
  区間の先頭・末尾は進捗 0・1、1 frame だけの区間は 1。step が一定の演出区間や段階間のリセットも扱えます。
- `Block` / `Edge` / `ArchitectureState` / `draw_architecture`: `xy` は左下座標。
  状態にない block は非表示、未知の ID・辺は例外です。`badges={id: style.CONCEPT_BADGE}` で
  「構想」バッジと破線の枠・辺を描きます。日本語には `Noto Sans CJK JP` を使います。
  `removed` は学習中の block の取り外しを想定し、学習中色を低彩度・不透明度20%にします。
  現在の状態 API は元色を保持しないため、他の状態からの取り外し遷移は例外にします。
- `CurveSeries` / `draw_progressive_curves`: step は昇順の `int64`、値は有限の `float64` 配列。
  カーソル以下の実測値と marker だけを表示します。head と色をそろえる場合は、呼び出し元で
  `ax.set_prop_cycle(color=[...])` を指定します。軸の範囲・ラベルと毎 frame の clear も呼び出し元が担当します。
- `render_presentation`: 正の偶数サイズの RGB frame を一度ずつ描き、`<name>.mp4`（H.264）、
  `<name>.gif`、`stills/<name>.png`、`manifest.json` を生成します。GIF は時刻に合わせて間引き・resize し、
  静止画は元解像度を維持します。既存の出力先は空でも拒否し、失敗時の部分出力も残します。
  manifest は呼び出し元で入力・checkpoint の SHA-256、git commit、解決済み config、
  `measured` / `schematic` / `concept` の由来をそろえ、JSON 化可能な値で渡します。

色と記法の正本は [`presentation/style.py`](presentation/style.py) です。
描画関数は matplotlib の global 設定や backend を変更しません。

## Web UIのGPU実行

GPU推論のHTTP要求は、元リポジトリの `.training_queue` に `resource=all` で
投入する。各要求は専用プロセスでモデルをロードし、推論終了時にGPUメモリを
解放する。HTTPサーバーはGPUモデルを保持せず、キューの完了後に結果を返す。
CPUを明示した要求はプロセス内で実行する。

待機中もカタログ・シーン閲覧は利用できる。キューの使用方法と取消操作は
[training-queue](../../../../.agents/skills/training-queue/SKILL.md)を参照。
要求JSONと結果は `.training_queue/ui_requests/`、実行ログは共有キューに保存する。
worktreeごとの独立したGPUキューは作成しない。
