# チケット: DINO Denoiser ラッパー実装

## 背景/目的
`third_party/DINO` 互換の denoising 生成をラップし、LightningModule の `training_step` から簡潔に利用可能にする。

## 対象
- `src/training/scene_model/dino_denoiser.py`

## タスク
- [ ] ノイズ付きクエリ/ラベル生成の API を設計（例: `make_noise(targets) -> dn_state`）。
- [ ] `configs/training/dino.yaml` の `num_noisy_queries`, `label_noise_scale` などとパラ連携。
- [ ] 生成物の形状・ID 整合のバリデーションを実装（assert/型ヒント）。
- [ ] 最小の単体テスト雛形を追加（件数/ID 整合）。

## 受け入れ基準（DoD）
- [ ] `make_noise` に対してモック `targets` を与えたときに期待する構造で返る。
- [ ] LightningModule から呼び出しても例外にならない。
