# WASB (Where's the Ball)

`src/wasb` は、テニス映像からのボール検出を行うタスク実装です。
動画からのフレーム抽出、ボール検出、クリップ分割、ラベル出力までを一連のパイプラインとして扱います。

## 目的 / 想定入出力

- **入力**: テニス映像（動画 or フレームシーケンス）
- **出力**: フレームごとのボール位置

## ディレクトリ構成

```
src/wasb/
├── configs/                          # Hydra 設定ファイル群
│   │
│   │ # 学習設定
│   ├── train_ball_detection.yaml     # ボール検出学習メイン設定
│   │
│   │ # データセット生成・可視化設定
│   ├── generate_dataset.yaml         # データ生成メイン設定
│   ├── download_videos.yaml          # 動画ダウンロード設定
│   ├── clip_sampling.yaml            # クリップサンプリング設定
│   ├── plot_ball_video.yaml          # ボール検出可視化設定
│   ├── plot_ball_video_ensemble.yaml # アンサンブル可視化設定
│   ├── save_one_sample_visuals.yaml  # サンプル確認設定
│   ├── extract_dinov3_backbone.yaml  # DINOv3 バックボーン抽出設定
│   ├── encode_dinov3_tokens.yaml     # パッチトークン事前計算設定
│   │
│   ├── run/                          # 実行時設定（タスク別）
│   │   └── ball_detection.yaml
│   │
│   ├── model/                        # モデルアーキテクチャ設定
│   │   ├── dinov3_heatmap.yaml       # DINOv3 ViT + Heatmap head
│   │   ├── dinov3_detr_heatmap.yaml  # DINOv3 + DETR デコーダ
│   │   ├── hrcnet.yaml               # HRCNet（時間方向 Conv）
│   │   ├── hrnet.yaml                # HRNet ベースライン
│   │   └── temporal_conv_gru.yaml    # Temporal Conv + GRU
│   │
│   ├── data/                         # DataModule 設定
│   │   ├── ball_detection.yaml
│   │   └── patch_embeddings.yaml     # 事前計算パッチ用
│   │
│   ├── training/                     # 学習ハイパーパラメータ
│   │   └── ball_detection.yaml
│   │
│   ├── loss/                         # 損失関数設定
│   │   └── ball_detection.yaml
│   │
│   ├── metrics/                      # 評価指標設定
│   │   └── ball_detection.yaml
│   │
│   ├── pipeline/
│   │   └── default.yaml
│   ├── download/
│   │   └── default.yaml
│   └── logging/
│       └── default.yaml
│
├── scripts/                          # 実行スクリプト（Hydra エントリポイント）
│   ├── train/                        # 学習スクリプト
│   │   └── ball_detection.py         # ボール検出モデル学習
│   │
│   ├── generate_dataset/             # データセット生成
│   │   ├── __main__.py               # エントリポイント（モード分岐）
│   │   ├── batch.py                  # バッチ処理（動画→アノテーション）
│   │   ├── clip_sampling.py          # クリッププレビュー・選別
│   │   └── download_videos.py        # YouTube 動画ダウンロード
│   │
│   ├── visualize/                    # 可視化スクリプト
│   │   ├── ball_video.py             # 単一モデル検出結果オーバーレイ
│   │   ├── ball_video_ensemble.py    # アンサンブル検出オーバーレイ
│   │   └── save_one_sample_visuals.py # データセットサンプル確認
│   │
│   └── tools/                        # ユーティリティ
│       ├── extract_dinov3_backbone.py # DINOv3 バックボーン抽出
│       └── encode_dinov3_patch_tokens.py # パッチトークン事前計算
│
├── models/                           # モデル実装
│   ├── ball_detection/               # ボール検出モデル
│   │   ├── dinov3_heatmap.py         # DINOv3 ViT + Heatmap デコーダ
│   │   ├── dinov3_detr_heatmap.py    # DINOv3 + DETR スタイル
│   │   ├── hrcnet.py                 # HRCNet（時間方向 Conv）
│   │   ├── hrnet.py                  # HRNet ベースライン
│   │   └── temporal_conv_gru.py      # Temporal Conv + GRU
│   │
│   └── others/
│       └── clip_segmenter.py         # クリップセグメンテーション
│
├── data/                             # データセット・DataModule
│   ├── ball_detection_dataset.py     # ボール検出 Dataset
│   ├── ball_detection_datamodule.py  # ボール検出 DataModule
│   ├── patch_embeddings_dataset.py   # 事前計算パッチ Dataset
│   ├── patch_embeddings_datamodule.py
│   └── curriculum_sampling.py        # カリキュラム学習サンプラー
│
├── training/                         # 学習関連
│   └── ball_detection/
│       ├── lightning_module.py       # ボール検出 LightningModule
│       ├── loss.py                   # Heatmap 損失（MSE、Focal 等）
│       └── metrics.py                # 検出精度指標
│
├── inference/                        # 推論
│   └── ball_detection/
│       ├── wasb_predictor.py         # WASB モデル推論
│       ├── hrcnet_predictor.py       # HRCNet 推論
│       └── heatmap_ensemble_predictor.py # アンサンブル推論
│
├── pipeline/                         # エンドツーエンドパイプライン
│   ├── annotation_pipeline.py        # 半自動アノテーション
│   └── video_ball_localization_pipeline.py # 動画からボール位置抽出
│
├── tennis_format.py                  # label.csv 等の I/O ヘルパー
│
├── utils/                            # ユーティリティ
│   ├── video_extractor.py            # 動画フレーム抽出
│   ├── streaming_loader.py           # ストリーミングローダ
│   └── checkpoint.py                 # チェックポイント操作
│
└── demo/
    └── wasb_ball_detection.py        # デモアプリ
```

## 主要コンポーネントの関係

```
┌─────────────────────────────────────────────────────────────────┐
│ データセット生成パイプライン                                     │
│   download_videos.py → batch.py → clip_sampling.py              │
│   └── → data/tennis/clips/                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 学習パイプライン                                                 │
│   scripts/train.py                                              │
│   ├── data/ball_detection_datamodule.py                         │
│   ├── models/ball_detection/*.py                                │
│   └── training/ball_detection/lightning_module.py               │
└─────────────────────────────────────────────────────────────────┘

## 手動アノテーション（クリップ作成）

動画全体ではなく、クリップを手動で作成してボール位置をアノテーションする場合は
`src.tools.annotate_wasb_clips` を利用します。クリップ作成 UI で範囲を指定し、
WASB が期待する `Clip*/Label.csv` 形式で保存します。

```bash
uv run python -m src.tools.annotate_wasb_clips mode=clip \
  video_path=data/raw/match.mp4 output.output_dir=data/tennis output.game_name=game_manual

# 既存の Clip を手動アノテーションする場合
uv run python -m src.tools.annotate_wasb_clips mode=annotation \
  output.output_dir=data/tennis output.game_name=game_manual annotate.clip_indices=[1,3]
```
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 推論パイプライン                                                 │
│   pipeline/video_ball_localization_pipeline.py                  │
│   └── inference/ball_detection/*_predictor.py                   │
└─────────────────────────────────────────────────────────────────┘
```

## Heatmap アンサンブル推論

`src/wasb/inference/ball_detection/heatmap_ensemble_predictor.py` は、複数モデルの
logit ヒートマップを TTA で生成し、逆変換で同一の `output_heatmap_hw` に整列した後、
温度校正 → TTA 平均 → PoE 融合 → forward-backward 平滑化を行います。最終的な座標は
平滑化後の分布から期待値/MAP/2次またはガウスフィットで復元します。

## 実行コマンド

詳細は [docs/scripts/wasb/](../../../docs/scripts/wasb/) を参照。

```bash
# データセット生成
uv run python -m src.wasb.scripts.generate_dataset

# 学習
uv run python -m src.wasb.scripts.train

# 可視化
uv run python -m src.wasb.scripts.visualize.ball_video
```
