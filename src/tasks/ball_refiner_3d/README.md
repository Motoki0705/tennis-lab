# 3D Ball Refiner

三角測量の3D座標 `(B,T,3)` と欠損mask `(B,T)`（true=欠損）から、
観測区間も含む全frameの3D軌道1本と、各frameのイベント確率を推論する。
オフラインの双方向Transformer。confidence・court・camera・GTイベントは入力しない。
2D Refinerは廃止した。2Dの補間点を三角測量へ渡さず、3Dで欠損を補完する。
シーンへの3D Refiner接続はこのtaskの学習・推論APIとは別の工程。

## モデルと損失

`models/generators/transformer.py` は共通componentsのRoPE / RMSNorm / SwiGLUを使う。
幅256・8層、FFNは8/3倍を64単位に切り上げた704。
座標ヘッドとイベント2logitヘッドを持ち、イベント確率はクラス軸のsoftmaxの第2成分。
時間軸をsoftmaxで正規化せず、複数イベントを表現する。
設定の正本は [model/coordinate_transformer.yaml](configs/model/coordinate_transformer.yaml)。

- 既定の直接回帰は、正規化3D座標の全frame **L1 + soft cross-entropy**。係数は位置:イベント=1:1。
- 3D座標の固定scaleはXYZそれぞれ10/20/5m。L1は軸・frame・batchの平均。係数1:1は実測loss値や勾配量の均等化を意味しない。
- イベント教師はshot/bounceを区別せず、イベント時刻を頂点1とするGaussian。複数イベントはmaxで合成する。ラリー全体で生成してから窓を切り出すので、窓外イベントの裾も残る。soft CEの教師は `[1-p,p]`。
- **GANとlossスケジュールは既定で無効**。位置lossは最後まで維持する。
- optional GANのDは出力3D軌道だけを評価する4層Transformer。位置への対応を保証する目的ではない。`training.gan.enabled=true` で有効化し、係数の段階増加はさらに `training.gan.schedule_enabled=true` を明示する。
- `model.architecture=flow` は条件付きx0予測Flow matchingの比較方式。既存の速度MSEを位置側の目的とし、同じ状態・時刻からイベントsoft CEも学習する。推論は明示seedから1本生成し、最終更新と同じforwardのイベント確率を返す。GANとは併用しない。

Gaussian幅・係数・optimizer・batch size・optional scheduleの正本は
[training/coordinate_gan.yaml](configs/training/coordinate_gan.yaml)。
G/DはAdamW。Generatorの学習率cosine減衰はloss係数のスケジュールとは独立。
入力の欠損値はforwardでゼロ化し、欠損frameも双方向attentionのqueryとして残す。

## データと拡張

共通データは既存の `data/ball_refiner/single_object/` を再利用する。
BLCSの3D軌道を一度だけ保存し、多視点投影・camera・イベントを同じrallyに保持する。
全cameraと窓はrally単位でtrain/val/testに分割し、モデルごとにデータを複製しない。
両baseline後方のfence面でcameraのX/Z・注視方向・画角を変える。
データ生成条件は [generate_coordinates.yaml](configs/generate_coordinates.yaml)。

拡張はイベント単位で連続欠損を選び、左右3〜10frameを異なる長さで抽選する。
選択イベントの欠損をcamera間で同期し、その2D観測だけから三角測量する。
2view未満・退化・負depthは3Dでも欠損。3D GTを入力へコピーしない。
既定はjitter・外れ値・離散欠損がすべて0。条件は [train_coordinates.yaml](configs/train_coordinates.yaml)。
欠損選択率やノイズを変更してアブレーション可能。評価条件とseedは固定し、入力hashを記録する。

```bash
# 生成先が既存なら停止する。今回の移行で再生成・削除はしない。
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.generate_coordinates
# CUDA実行は下記の共有training queueから登録する。
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.train_coordinates
```

ローカルGPUは必ず[training queue](../../../.agents/skills/training-queue/SKILL.md)を使う。
既定出力は `outputs/ball_refiner_3d/train/coordinates/<run>/`。
validation全frame位置RMSEでbestを選び、best/last両方を固定testで評価する。
`logs/version_0/checkpoints/{best,last}.ckpt` と `predictions*/pred_test.npz` に保存し、
NPZには座標・欠損・frame/rally対応に加えて `event_probability` と `event_target` を保持する。
位置指標に加えてイベントBrier scoreとsoft CEを記録する。位置RMSEはm単位のユークリッド距離。

## 推論・レビュー

```bash
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.predict_coordinates \
  --checkpoint /absolute/run/logs/version_0/checkpoints/best.ckpt \
  --input /absolute/input.npz --output /absolute/prediction.npz --device cpu
```

入力NPZは `coordinates`, `missing`, `fps` のみ。座標はm、欠損はbool。
FPSはcheckpointと一致させる。出力は同じ単位の座標とframeごとの `event_probability`。
重複窓では座標とイベントを同じ窓から採用し、短い系列は実frameだけで処理する。
checkpoint schema `ball_refiner_3d.events.v1` はGaussian幅とイベントヘッドを必須とし、
旧2D/GMM/イベントヘッドなし3D重みは明示的に拒否する。過去のデータ・重み・knowledgeは保存する。
旧実験の再現にはその記録のcommitを使用する。

[Dataset Review](review/README.md)ではcheckpoint候補、拡張前後、3D推論とGT、イベント確率とGaussian教師を比較できる。
