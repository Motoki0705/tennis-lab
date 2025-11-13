# SceneModel Training 仕様（概要）

本ディレクトリは、`src/training/scene_model/` とその周辺（`src/datasets/`、`src/models/scene_model/`、`src/training/utils/`）に基づく学習パイプラインの仕様をまとめる。コードを読まずに役割・入出力・ワークフローを把握できることを目的とする。

## ワークフロー（擬似コード）

```
# 前提: cfg は DictConfig（YAML由来）。
loader = ConfigLoader(cfg)
dm      = loader.build_datamodule()       # DancetrackDataModule
lit     = loader.build_lit_module()       # SceneModelLightningModule
logger  = loader.build_logger()           # TensorBoardLogger
cbs     = loader.build_callbacks()        # Checkpoint, LR monitor など
trainer = loader.build_trainer(logger, cbs)

trainer.fit(lit, datamodule=dm)
```

学習ループ（バッチ毎）:

```
batch = SceneBatch(
  frames: Tensor[B, T, 3, H, W],
  targets: list[list[TargetFrame]],   # 長さ T（パディング後）の各フレーム注釈
  padding_mask: Bool[B, T],           # True=pad, False=valid
  sequence_ids: list[str]
)

outputs = model(frames)  # SceneModel
# 代表例（BBoxHead）
#   bbox_center: Float[B, T, Q, 2]
#   bbox_size:   Float[B, T, Q, 2]
#   cls_logits:  Float[B, T, Q, C]
#   exist_conf:  Float[B, T, Q, 1]

dn_state = denoiser.make_noise(targets)   # DINO風ノイズ（現状 loss 未使用）
losses   = head_adapter.compute_loss(outputs, targets, padding_mask, dn_state)
  # losses = {
  #   total, cls, bbox, giou, exist,
  #   num_matches, num_targets
  # }
backprop(losses.total)
```

検証時は loss の記録に加えて、先頭バッチの先頭サンプルから最大 `viz.max_frames` フレーム分の可視化グリッドを TensorBoard に出力する。

## 主要コンポーネントと役割

- `DancetrackDataModule`（データ）
  - `DanceTrack` ウィンドウ化データセットを構築し、`collate_tracking` で `[B,T,...]` にパディングする。
  - バッチは `SceneBatch` で受け渡される。

- `SceneModelLightningModule`（学習制御）
  - `build_scene_model(cfg.model, cfg.debug)` で SceneModel を組み立て、損失は `HeadAdapter` へ委譲。
  - 最適化器・スケジューラ、バックボーン凍結/解凍、ロギング、可視化を管理。

- `HeadAdapter`（損失）
  - ハンガリアンマッチング（DINO 実装または簡易フォールバック）で予測と GT を対応付け、`cls / exist / bbox(L1) / giou` を加重合算。

- `DinoDenoiser`（ノイズ）
  - DINO 風に GT ボックス・ID にノイズを与える補助入力を生成（現状は損失未使用）。

- `callbacks`（ロガー/コールバック）
  - TensorBoard ロガー、ModelCheckpoint、LearningRateMonitor を設定から構築。

- `ConfigLoader`（設定読込）
  - YAML を階層マージし、上記オブジェクト群をファクトリ的に生成。

## テンソル形状の要約

- 画像: `frames` は `[B, T, 3, H, W]`（正規化済み）。
- パディング: `padding_mask` は `[B, T]`（pad=True）。
- 予測（BBoxHead）：
  - `bbox_center`: `[B, T, Q, 2]`（cx, cy）
  - `bbox_size`:   `[B, T, Q, 2]`（w, h; 0〜1 正規化、`sigmoid` → `clamp(min=1e-3)`）
  - `exist_conf`:  `[B, T, Q, 1]`（ロジット→Sigmoid後）
  - `cls_logits`:  `[B, T, Q, C]`
- GT（TargetFrame per frame）：
  - `center`: `[N, 2]`, `size`: `[N, 2]`（いずれもリサイズ後キャンバスに対する 0〜1 正規化）, `track_ids`: `[N]`

## バックボーン凍結ポリシー

- `cfg.model.backbone.freeze: bool` が True の場合、初期エポックで `requires_grad=False`。
- `cfg.model.backbone.unfreeze_after: int` 以降で自動的に解凍。

## 可視化（検証）

- `val` の先頭バッチで、上位スコア or 閾値超えの予測ボックスをフレームに描画し、`make_grid` で横並び出力。

