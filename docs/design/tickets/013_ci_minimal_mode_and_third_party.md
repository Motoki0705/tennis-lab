# チケット: CI最適化（最小モード/third_party 取り回し）

## 背景/目的
dinov3/DINO 依存によりビルドが重くなるため、CIやローカル検証用に最小モード（モック重み/小入力）を用意する。

## 対象
- `configs/models/scene_model.yaml`（最小モード切替）
- `src/models/scene_model/build.py`（モック重み/スキップ処理）
- `tests/`（マーカーや環境変数でスキップ）

## タスク
- [ ] `CFG_DEBUG_MINIMAL=true` 環境変数または `cfg.debug.minimal=true` で軽量構成へ切替（小ViT/小解像度/短window）。
- [ ] dinov3 重みが見つからない場合のフォールバック（ランダム初期化+警告）。
- [ ] テスト時は third_party 参照をモック化（import ガード/ダミークラス）。
- [ ] ドキュメントに最小モードの利用方法を追記。

## 受け入れ基準（DoD）
- [ ] 最小モードで `uv run pytest -q` が 1 分以内に完了。
- [ ] 重み未配置でも学習初期化が可能（ただし性能は不問）。
