# Shared Visualization

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
