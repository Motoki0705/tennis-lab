# 学習環境アーキテクチャ (Training Environment Architecture)

本プロジェクトでは、保守性と拡張性を高めるために共通の学習環境アーキテクチャを採用しています。

## 1. ディレクトリ構造

以下の共通したディレクトリ構造を持ちます：

- `configs/`: 学習およびモデルの設定ファイル (YAML)
- `data/`: データセット、DataModule、データ生成ロジック
- `models/`: モデルアーキテクチャの定義
- `training/`: PyTorch Lightning Module、損失関数、評価指標
- `scripts/`: 学習・評価・データ生成用のスクリプト

## 2. フレームワークと技術スタック

- **PyTorch Lightning**: 学習ループの標準化 (`pl.LightningModule`)
- **Omegaconf**: YAMLベースの設定管理
- **Optimizer**: `AdamW` を標準採用
- **Scheduler**: ウォームアップ (`LinearLR`) とコサインアニーリング (`CosineAnnealingLR`) を組み合わせた `SequentialLR`

## 3. 設定管理 (Configuration)

設定ファイル (`default.yaml`) は以下のセクションで統一します：

- **model**: モデルのハイパーパラメータ (次元数、レイヤー数など)
- **data**: データセットパス、バッチサイズ、分割比率、Augmentation設定
- **training**: 学習率、エポック数、損失関数の重み
- **metrics**: 精度評価の閾値

## 4. 実装の共通パターン (LightningModule)

`training/lightning_module.py` は以下の共通パターンで実装します。

- **`__init__`**: 設定ファイルからのパラメータ読み込み、モデル・損失関数・メトリクスの初期化
- **`_shared_step`**: Train/Val/Test で共通の処理（Forward pass, Loss計算, Metrics更新）を集約
- **`configure_optimizers`**: OptimizerとSchedulerの構築ロジックを統一

## 5. 関連ドキュメント

- [Base モジュール概要 (`src/base`)](./core/base.md)
- [Utils モジュール概要 (`src/utils`)](./core/utils.md)
