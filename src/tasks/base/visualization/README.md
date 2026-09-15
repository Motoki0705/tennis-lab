# Shared Visualization

## タスク別の使い方

起動コマンドと操作手順は各ガイドを正本とします。

| タスク | ガイド | 閲覧 / 推論ポート |
|---|---|---|
| PLCS | [利用ガイド](../../plcs/visualization/README.md) | 8772 / 8771 |
| BLCS | [利用ガイド](../../blcs/visualization/README.md) | 8773 / 8770 |
| Court Detection | [利用ガイド](../../court_detection/visualization/README.md) | 8774 / 8775 |
| Ball Detection | [利用ガイド](../../ball_detection/visualization/README.md) | 8776 / 8777 |

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

## Web UIのGPU実行

GPU推論のHTTP要求は、元リポジトリの `.training_queue` に `resource=all` で
投入する。各要求は専用プロセスでモデルをロードし、推論終了時にGPUメモリを
解放する。HTTPサーバーはGPUモデルを保持せず、キューの完了後に結果を返す。
CPUを明示した要求はプロセス内で実行する。

待機中もカタログ・シーン閲覧は利用できる。キューの使用方法と取消操作は
[training-queue](../../../../.agents/skills/training-queue/SKILL.md)を参照。
要求JSONと結果は `.training_queue/ui_requests/`、実行ログは共有キューに保存する。
worktreeごとの独立したGPUキューは作成しない。
