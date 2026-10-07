# 3D Ball Refiner

三角測量した3D座標 `(B,T,3)` と欠損mask `(B,T)`（true=欠損）から、
観測区間を含む全frameの軌道1本と、各frameのイベント確率を推論する。
オフラインの双方向Transformer。confidence・court・camera・GTイベントはモデルへ渡さない。

## 構成と依存

| ディレクトリ | 責務 |
|---|---|
| `configs/` | Hydraのdata・augmentation・model・loss・training・generation・visualization設定 |
| `configuration/` | 設定の厳密な検証、path境界、旧実験設定の明示的な読み取り |
| [data/](data/README.md) | NPZ読込、拡張、三角測量、イベント教師、窓サンプリング、DataModule |
| `generate_dataset/` | BLCS物理生成adapter、camera選択、保存、既存3D軌道の再投影 |
| [models/](models/README.md) | 共通trunk、回帰・Flow generator、軌道discriminatorのforward |
| `model_io/` | 入出力型、正規化、adapter、factory、checkpoint読込 |
| [training/](training/README.md) | BaseTrainingRunner / BaseLightningModuleへの接続、loss、step schedule |
| `inference/` | 公開Predictor、窓の統合、Flow sampling、NPZ入出力 |
| `evaluation/` | 全ラリー評価、baseline、checkpointと対応付けた予測・指標の保存 |
| [visualization/](visualization/dataset_review/README.md) | 静的plotとDataset Review WebUI |
| `scripts/` | 引数・path検証と上記APIの呼び出しのみ |

`models` はloss・sampling・dataset・WebUIを参照しない。
`model_io` がforward前にshape/device/maskを検証し、共通 `BoundModelIO` を通じてモデルを呼ぶ。
汎用Transformer部品は `src/utils/models/components`、学習基盤は `src/tasks/base` を再利用する。
シーンへの3D Refiner接続はこのtaskの学習・推論APIとは別の工程。

## 実行

```bash
# 既存datasetがあれば停止する。共有データの再生成は通常不要。
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.generate_dataset

# ローカルCUDAは必ずtraining queueへ登録して実行する。
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.train
# Flowの比較実験は model=flow。回帰は model=regression。
```

GPU実行は[training queue](../../../.agents/skills/training-queue/SKILL.md)に従う。
学習設定・再開・保存先は[training README](training/README.md)を参照。

```bash
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.predict \
  --checkpoint /absolute/run/logs/version_0/checkpoints/best.ckpt \
  --input /absolute/input.npz --output /absolute/prediction.npz --device cpu

.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.evaluate \
  --checkpoint /absolute/run/logs/version_0/checkpoints/best.ckpt \
  --run-config /absolute/run/config.yaml \
  --dataset /absolute/data/ball_refiner/single_object \
  --output /absolute/new-evaluation --device cpu

.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.review_dataset
```

入力NPZは `coordinates`, `missing`, `fps` のみ。座標はm、欠損はbool。
公開API `inference.RefinerPredictor` は同じ契約で入力し、CPU上の座標とイベント確率を返す。
FPSはcheckpointと一致させる。座標と確率は重複窓の同じ担当窓から採用する。
新しい評価先・推論出力は既存ファイルを上書きしない。

## 互換性

既存のイベントヘッド付き `ball_refiner_3d.events.v1` 重みは、そのまま推論・初期重みとして使える。
新しい学習はoptimizer・scheduler・乱数状態を持つLightning `events.v2` checkpointを保存する。
2D/GMM/イベントヘッドなし3Dの重みは明示的に拒否する。
共有dataset・学習済み重み・過去の実験記録は変更しない。
旧CLI・flat moduleは廃止したため、過去実験の再実行には記録されたcommitを使用する。
