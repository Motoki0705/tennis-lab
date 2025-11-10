# DanceTrack を用いた SceneModel 学習環境設計（Lightning 版）

## 1. 目的・スコープ・非目標
- **目的**: `third_party/DanceTrack/dancetrack` を活用し、テニス領域向け `SceneModel` の潜在能力を検証するための Lightning ベース学習環境を構築する。`third_party/dinov3` の ViT-S と `third_party/DINO` の denoising 戦略を取り込み、bbox トラッキングタスクでヘッドを検証できるようにする。
- **スコープ**: データパイプライン（`src/datasets`）、Lightning DataModule/Module（`src/training/scene_model`）、モジュラー設定（`configs/` + `src/training/utils/config.py`）、CLI（`src/cli`）、`SceneModel` ビルド（`src/models/scene_model/build.py`）、TensorBoard ログ。
- **非目標**: DanceTrack 以外のデータ統合、推論 CLI、分散学習/マルチノード最適化、テニス本番ヘッドへの回帰実装。

## 2. 背景/課題と前提・依存
- SceneModel は 3D/役割推論器として設計済みだが教師データが不足。DanceTrack で bbox/ID 追跡に対応させ、幹部（ViT/Temporal）の事前学習を行いポテンシャルを測る。
- ViT 部は既存のランダム初期化では収束が遅いため、`third_party/dinov3` の pretrained ViT-S 重みを Backbone に差し替える。`src/models/scene_model/build.py` で `dinov3_vits` を取り込めるようにする。
- 早期収束のため `third_party/DINO` の denoising ブロックを損失計算に導入し、ノイズ付き bbox/ID をペアリングする。
- 依存: `uv`、PyTorch 2.2+、PyTorch Lightning 2.x、torchvision v2 transforms、TensorBoard、`third_party/DanceTrack`、`third_party/dinov3`、`third_party/DINO`。単一 GPU 想定。

## 3. 要求/制約
1. **ディレクトリ構成（モジュラー構成に更新）**
   ```
   configs/
     datasets/dancetrack.yaml      # Dataset 専用設定
     training/dino.yaml            # 学習ループ/最適化設定
     models/scene_model.yaml       # モデル/ヘッド構成
     logging/default.yaml          # ロギング/ckpt 設定
     scene_model.yaml              # 上記4つを参照するトップレベル

   src/
     cli/train_scene_model.py
     datasets/dancetrack.py
     datasets/collate_tracking.py  # 旧 collate.py から明確化
     models/scene_model/build.py   # dinov3 backbone / head 切替

   training/
     utils/config.py               # YAML 合成/CLI 上書き
     scene_model/
       lightning.py                # 旧 lit_module.py → lightning.py
       datamodule.py               # LightningDataModule
       head_adapter.py             # DINO Hungarian 利用のマッチング
       dino_denoiser.py            # third_party/DINO 互換ノイズ生成
       callbacks.py                # ckpt, lr-monitor, tensorboard
   ```
2. **設定運用**: `training/utils/config.py` が
   - `configs/scene_model.yaml` を入口に、`datasets/training/models/logging` の各 YAML を読み込み、
   - 単一の `cfg`（`DictConfig`）に合成、
   - `src/cli/train_scene_model.py` の CLI 引数（例: `--training.trainer.max_epochs 30`）でドット表記上書き。
3. **Lightning 採用**: 学習ループは `pl.Trainer`、`LightningModule`、`LightningDataModule` で統一。AMP/Checkpoint/Log は Lightning の標準機能を利用。
4. **Transforms**: 画像前処理は `torchvision.transforms.v2`（`Resize`、`ConvertImageDtype`、`Normalize`、`RandomHorizontalFlip` 等）に限定し、独自 transforms は作らない。
5. **TensorBoard**: 進捗は `pytorch_lightning.loggers.TensorBoardLogger` で `/runs/dancetrack_scene_model` に出力。`uv run tensorboard --logdir runs` を想定。
6. **実行コマンド**: `uv run python -m src.cli.train_scene_model --config configs/scene_model.yaml --data.root third_party/DanceTrack/dancetrack`。
7. **テスト**: `tests/training/test_dancetrack_datamodule.py` 等で LightningDataModule の smoke を行う。

## 4. アーキテクチャ方針
| 案 | 内容 | Pros | Cons |
| --- | --- | --- | --- |
| A. PyTorch Lightning | LightningModule/DataModule/Trainer を採用。TensorBoard, ckpt, AMP を内包。 | CLI/API 一貫、監視/再現性が高い。 | 依存追加・Lightning 流儀に合わせた設計が必要。 |
| B. カスタム Trainer | 既存 PyTorch ループを継続。 | 依存最小。 | 監視/再開/CLI パラ合流をすべて自作。 |

→ 本計画では **案A**（Lightning）を採用。CLI と YAML の統合、TensorBoard ログ、DINO ノイズ注入を Lightning Hooks で安全に扱えるため。

## 5. モジュール分割/責務
- `src/datasets/dancetrack.py`
  - DanceTrack シーケンスを読み、`torchvision.io.read_image` + v2 transforms で `[T,3,H,W]` を生成。
  - bbox/ID を `TargetFrame`（center/size/track_id/conf）へ整形。
- `src/datasets/collate_tracking.py`
  - 可変長 window をパディングし、`SceneBatch(frames, targets, masks)` を返却。トラッキング用途と明示するため命名を変更。
- `src/training/scene_model/datamodule.py`
  - `LightningDataModule`。`setup(stage)` で train/val Dataset を生成。
  - シーケンスメタをキャッシュ（JSON）して IO 負荷を軽減。
- `src/training/scene_model/head_adapter.py`
  - `SceneModel` デコーダ出力を bbox/ID ロスへ変換。
  - マッチングは再実装せず、`third_party/DINO/models/dino/matcher.py` の `HungarianMatcher` を直接利用。
  - `exist_conf` gating、`bbox_delta` 予測、ID ロス（CE）、IoU ロスを提供。
  - 例: `from third_party.DINO.models.dino.matcher import HungarianMatcher` を用い、`matcher(outputs, targets)` の返り値（インデックス組）をそのままロスに適用。
- `training/scene_model/dino_denoiser.py`
  - `third_party/DINO` の denoising 実装をラップし、ノイズ付きクエリ/ラベルを生成。
  - LightningModule の `training_step` で `adapter.compute_loss(decoded, targets, dn_state)` を呼び出す。
- `src/training/scene_model/lightning.py`
  - `SceneModelBuilder`（後述）と optimizer/scheduler/TensorBoard logging を司る。
  - `configure_optimizers` で `AdamW + cosine` を返す。
- `src/training/utils/config.py`
  - `ConfigLoader.from_yaml(path, cli_args)` で dataclass を返却。CLI から `--config`/`--set key=value` を受けて統合。
  - Lightning Trainer/DataModule/Module の初期化オブジェクトを生成するファクトリを持つ。
- `src/models/scene_model/build.py`
  - `build_scene_model(cfg)` が `dinov3` Backbone、TemporalEncoder、`bbox head`、`DINO denoiser` 用 hooks を束ねて返す。
  - `dinov3` Backboneは以下のようにロードする。またコンフィグによってパラメータの凍結やウォームアップ後に回答などの設定ができるようにする。
    ```python
    dinov3_vits16 = torch.hub.load("third_party/dinov3", 'dinov3_vits16', source='local', weights="third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    ```
  - `cfg.head.type` に応じて従来 head or bbox head を切り替え。
- `src/cli/train_scene_model.py`
  - `--config` 指定で `training/utils/config.py` を呼び、`pl.Trainer` と Lightning モジュールを初期化→`trainer.fit()`。
- `configs/*.yaml`
  - `configs/scene_model.yaml` が `datasets/training/models/logging` の各 YAML を参照し、単一 cfg に合成。
  - 選択的に入れ替え可能（例: `dataset` を COCO へ変更、`training` を DINO→DETReg へ変更）。

## 6. データモデル/主要フロー
1. **設定ロード**
   ```python
   from training.utils.config import ConfigLoader
   from omegaconf import OmegaConf
   from src.training.utils.config import load_cfg
   cfg = load_cfg("configs/scene_model.yaml")
   # CLI 上書きは argparse + OmegaConf.merge を想定
   trainer = cfg.build_trainer()             # pl.Trainer
   lit_module = cfg.build_lit_module()      # SceneModelLightningModule
   datamodule = cfg.build_datamodule()
   ```
2. **SceneModel ビルド**
   - `src/models/scene_model/build.py` で
     - `third_party/dinov3` から ViT-S backbone をロード（`state_dict` で patch embedding & transformer を注入）。
     - Temporal/decoder/positionalは既存実装を再利用。
     - `head_adapter` で bbox head を組み込む（SMPL 出力を 0 次元に設定）。
3. **データ取得**
   - `LightningDataModule` が `src/datasets/dancetrack.py` を使い
     - `torchvision.transforms.v2.Compose([Resize, ConvertImageDtype, Normalize, RandomHorizontalFlip(p=0.5)])`
     - `window_sampler`: `DINO` の denoising で必要な `num_noisy_queries` をメタ情報から計算。
4. **学習ステップ（LightningModule）**
   ```python
   def training_step(self, batch, batch_idx):
       dn_targets = self.denoiser.make_noise(batch.targets)
       outputs = self.model(batch.frames)
       loss_dict = self.head_adapter.compute_loss(
           outputs, batch.targets, dn_targets
       )
       self.log_dict(loss_dict, prog_bar=True, logger=True)
       return loss_dict["total"]
   ```
   - DINO denoiser でノイズ付き bbox/ID を生成、Lightning が自動で AMP/grad-clip を実行。
5. **TensorBoard ロギング**
   - `TensorBoardLogger(save_dir="runs", name=cfg.experiment_name)`。
   - `training_step`/`validation_step` で `self.log("metrics/idf1", value)` などを出力。
6. **実行**
   ```
   uv run python -m src.cli.train_scene_model \
     --config configs/scene_model.yaml \
     --set dataset.root=third_party/DanceTrack/dancetrack training.trainer.max_epochs=30
   ```

## 7. トレードオフ/リスク/未決事項
- **依存増加**: Lightning + dinov3 + DINO によりビルド時間が増加。CI では minimal モード（mock weights）を用意する必要。
- **ヘッド切替の整合性**: bbox head と既存 3D head の共存は `SceneModelBuilder` での条件分岐が複雑化。Config テストを追加しないと regressions の恐れ。
- **DINO denoising の適用度**: DanceTrack 専用にチューニングされたノイズスケールがないため、`lambda_dn` 調整を実験で決める必要がある。
- **ViT-S への置換**: dinov3 の入力解像度規約（224×224）が SceneModel の `img_size` と異なる場合、位置埋め込み補間を慎重に実装する必要。
- **Lightning 依存**: 既存ユニットテストが純 PyTorch を想定している箇所はアップデートが必要。

## 8. 検証計画
1. **Dataset/Transforms Smoke**: `uv run pytest tests/training/test_dancetrack_datamodule.py -k smoke` で DataModule が 2 バッチ生成できるか確認。
2. **Lightning Module テスト**: `tests/training/test_scene_model_lit_module.py` で forward/backward を 1 ステップ確認し、TensorBoard ログファイルが生成されるか検証。
3. **DINO Denoiser テスト**: `tests/training/test_dino_denoiser.py` でノイズ付きクエリ数や ID 整合を確認。
4. **指標**: DanceTrack Val で HOTA / IDF1（TensorBoard に記録）。3epoch 以内で baseline ByteTrack の 60% を超えることを早期収束指標とする。
5. **テニスデータ転移**: 事前学習済み重みを別 config でロードし、テニスデータの validation loss が未学習モデルより低いことを確認（別タスクで実施）。

## 9. 参照/変更履歴
- 参照: docs/design/models/scene_model.md, docs/spec/models/scene_model.md, third_party/DanceTrack/README.md, third_party/dinov3/, third_party/DINO/
- 変更履歴:
  - 2025-02-17: Lightning/DINO/dinov3 対応案へ更新。
