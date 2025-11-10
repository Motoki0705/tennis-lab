# チケット: スモーク/ユニットテスト追加（DataModule/Lightning/Denoiser）

## 背景/目的
実装の回帰を防ぎ、最小限の信頼性を担保するため、スモークテストとユニットテストを整備する。

## 対象
- `tests/training/test_dancetrack_datamodule.py`
- `tests/training/test_scene_model_lit_module.py`
- `tests/training/test_dino_denoiser.py`

## タスク
- [ ] DataModule: 2バッチ分を取得できるスモーク（小さな window/resize を設定）。
- [ ] LightningModule: ダミー入出力で 1 step forward/backward（optimizer step 含む）を確認。
- [ ] Denoiser: ノイズ付きクエリ数・ID 整合のチェック。
- [ ] `uv run pytest -q` がローカルで通るように third_party 依存をモック or スキップ制御（マーカー）を実装。

## 受け入れ基準（DoD）
- [ ] 3本のテストファイルがグリーン。
- [ ] CIや低リソース環境でもタイムアウトせず完了。
