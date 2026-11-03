# Player Detection

出力先は [タスク出力規約](../OUTPUTS.md) に従います。

bboxの予測・修正、追跡/IDレビュー、pose公開、検出器の追加学習をつなぐ変更後設計は、
[注釈・継続学習のアーキテクチャ](../base/annotation/README.md)を参照してください。
実装状況は同資料からリンクしたサブIssueで管理します。

chat-annotation の処理済み player ラベルから、テニスプレーヤー（主コート上の選手）の検出器を作ります。
公式 DINO 4-scale Swin-L の COCO person logit（class id 1）を「tennis player」として fine-tune し、
`DinoPersonDetector` がそのまま読める DINO 形式の checkpoint を export します。
pipeline 側は `people_models.dino_checkpoint` を差し替えるだけで使えます。

新規clipへの予測案とレビューの接続は [annotation](annotation/README.md) を参照してください。
修正bboxからの[追跡・ID・pose](annotation/pose/README.md)と、
確認済み負例を含む[累積bbox公開](generate_dataset/publishing/README.md)を接続しています。
追加学習前の既存データ固定とvalidation/testの支持対象は
[snapshot・固定partition](data/snapshots/README.md)で管理します。
公開後の追加学習・中断再開・固定validation比較・モデル採否は
[annotation learning adapter](training/annotation/README.md)を参照してください。

## 流れ

1. `scripts.generate_dataset`: `outputs/chat_annotation` の processed player ラベルから
   `data/player_detection/<version>/` のv2 frame storeを作ります。空frameを含めnative frameを保存し、
   確認済み不在を`confirmed_empty`で区別します。clip-localのtrack IDと時刻も保持します。
   split の単位は YouTube の source 動画です。同じ version への上書きは拒否します。
   保存形式と読み出し API の正本は [data/store.py](data/store.py) です。
2. `scripts.review_dataset`: 保存済みRGB、注釈の出自・未解決状態、学習選別をWebで確認します。
   [レビュー画面の使い方とデータ体系](review/README.md)を参照してください。
   `scripts.preview_dataset` は従来のsplitごとの静的frame previewを出力します。
3. `scripts.train`: Lightning で fine-tune します。frame の選別（未 review、`unresolved` player を含む frame、
   画面外にほぼ出たboxの除外）は[data/selection.py](data/selection.py)が確認画面と共有し、
   除外件数を split ごとに表示します。
   確認済み不在はbbox/labelが0件の負例です。未確認・context・未解決やclip後のbox消失を負例にはしません。
4. `scripts.export_checkpoint`: Lightning checkpoint を DINO 形式の `.pth` に変換します。
   変換前に上流 DINO へ strict load して構造を確かめ、出自を `tennis_lab` キーに記録します。
5. `scripts.evaluate`: COCO 版と export 版など複数の checkpoint を、pipeline と同じ前処理で比較します。

設定の正本は [configs/](configs/) です。評価の `input_size` と `score_threshold` は、
pipeline の `people_models.runtime.dino_detector` と一致させる必要があります。
DINO の denoising は CUDA 前提のため、学習と評価は GPU（training queue 経由）で実行します。
学習前のLightning sanity checkは部分検証のため、mAPを計算・記録せず評価状態をリセットします。
validation全体に正例があれば、先頭のsanity対象batchが負例のみでも学習を開始できます。
本評価（validation/test）のmAPには正例が必要です。負例だけの場合は未定義値を選択指標にせず停止します。
mixed評価では負例上の検出もFP/APへ反映し、負例画像数とGT instance数を記録します。

通常readerはv2専用です。v2の空frameにはnative JPEGと
`(0,4)`のbbox・`(0,)`のlabelを使います。schema、確認済み不在フラグの整合性、instance範囲を
readerで検証します。全空/混在batchのCPU matcher/loss検証とCUDA受入の範囲は
[累積bbox公開](generate_dataset/publishing/README.md)に記載しています。

既存v1 storeを使う場合は、次の明示的な移行を先に実行し、出力されたv2を指定します。
sourceとdestinationは絶対パスで指定し、destinationはsourceの外にある未作成ディレクトリにします。
元snapshotは変更せず、split、保存済みframe行・時刻、JPEG bytes、教師を維持します。
欠落frameは欠落のまま残し、全保存行の`confirmed_empty`はfalseです。負例は追加しません。
既存destinationや不正なv1は拒否し、v2の検証完了後に新storeを公開します。

```bash
.venv/bin/python -m src.tasks.player_detection.scripts.migrate_store \
  /absolute/path/to/v1-store /absolute/path/to/new-v2-store
```

通常の学習・評価・静的プレビューは、DATA root内の`player_detection/chat-player-v2`を既定で読みます。
生成先の既定版も`chat-player-v2`です。既にこのstoreがある場合は生成を省いて利用します。
生成コマンドは既存storeへの上書きを拒否します。

```bash
.venv/bin/python -m src.tasks.player_detection.scripts.train
.venv/bin/python -m src.tasks.player_detection.scripts.preview_dataset
.venv/bin/python -m src.tasks.player_detection.scripts.export_checkpoint \
  export.lightning_checkpoint=<output-root 相対 .ckpt> \
  export.destination=player_detection/<name>.pth
.venv/bin/python -m src.tasks.player_detection.scripts.evaluate \
  '+evaluate.checkpoints.player_ft=player_detection/<name>.pth'
```

新しい版を生成するときは、未使用の`dataset.version`を明示します。
そのstoreを利用する各コマンドには、対応するdataset overrideを指定します。
別の公開済みv2 storeを使う場合も同じoverrideで選択できます。

```bash
.venv/bin/python -m src.tasks.player_detection.scripts.generate_dataset \
  dataset.version=<new-version>
.venv/bin/python -m src.tasks.player_detection.scripts.train \
  data.dataset_dir=player_detection/<new-version>
.venv/bin/python -m src.tasks.player_detection.scripts.evaluate \
  evaluate.dataset_dir=player_detection/<new-version>
.venv/bin/python -m src.tasks.player_detection.scripts.preview_dataset \
  preview.dataset_dir=player_detection/<new-version>
```

DINO の構築と前処理は [src/submodules/models/dino/architecture.py](../../submodules/models/dino/architecture.py) を
推論（`DinoPersonDetector`）と共有します。

## 学習中の定性保存

`training.qualitative_logging.enabled=true`が既定で、選択されたvalidation batchの
先頭sampleに対応する元動画を、学習中のモデルで推論してbbox・confidence付きGIFに保存します。
同じクリップが複数batchから選ばれた場合は1回だけ描きます。
選択batch数は`num_samples`、保存周期は[共通契約](../base/README.md#qualitative-logging)に従います。

入力動画は`qualitative.annotation_root`（OUTPUT root相対、既定`chat_annotation`）の
prepared manifestと公開済み動画から解決します。frame storeと動画hash・寸法・frame数を照合し、
元動画のPTS順を検証してdecodeするため、storeにないframeやplayer不在frameも推論対象です。
manifest・元動画が欠落／不一致なら停止します。

`qualitative.frame_stride`ごとにクリップ先頭から最大`qualitative.max_frames`枚を描き、
推論batch sizeは`qualitative.batch_size`、表示幅の上限は`qualitative.display_width`です。
再生FPSは元動画の公称FPSをframe strideで割ります。
値の正本は[qualitative/default.yaml](configs/qualitative/default.yaml)。
推論の前処理と閾値は`data.input_size`と`evaluation.score_threshold`／`max_detections`を共有します。
出力は共通ログ配下の`qualitative/epoch_XXXX/player_clipNNNN.gif`とTensorBoardです。
`NNNN`はframe storeのclip indexで、GIF内の見出しにもclip ID・元frame indexを記載します。
このGIFにはGT boxやtracking IDは付与しません。

`visualization/qualitative.py`が元動画の解決とクリップ推論、`visualization/overlays.py`が
評価と共用するbbox描画を所有します。定性保存を無効にする場合は
`training.qualitative_logging.enabled=false`を指定します。

## fine-tuned checkpoint での推論

`inference.DinoPlayerDetector` は export 済みの player checkpoint の出自を検証してから、
既存の `DinoPersonDetector` でロード・推論します。COCO の person checkpoint や
Lightning の `.ckpt` を player 用として誤って読まないための入口です。
入力は BGR `uint8` frame、出力は元画像の pixel `xyxy` box と confidence です。
tracking ID は検出器の出力には含まれません。

今回の学習で validation AP が最高だった epoch 3 の Lightning checkpoint は
`ckpt/player_detection/chat-player-v1-e8-best-epoch03.ckpt`、推論用に export した
DINO 形式は `ckpt/player_detection/chat-player-v1-e8-best-pr937.pth` です。
推論には後者を使います。pipeline では
`people_models.dino_checkpoint=player_detection/chat-player-v1-e8-best-pr937.pth`
を指定できます。

```python
from pathlib import Path

from src.tasks.player_detection.inference import (
    DinoPlayerDetector,
    PlayerDetectionRequest,
)

root = Path("/path/to/tennis-lab")
detector = DinoPlayerDetector(
    root / "ckpt/player_detection/chat-player-v1-e8-best-pr937.pth",
    root / "third_party/DINO",
    device="cuda",
    confidence=0.3,
    short_side=800,
    max_long_side=1333,
)
result = detector.predict(PlayerDetectionRequest(frame_bgr=frame_bgr))
detector.unload()
```

`frame_bgr` は OpenCV などで読み込んだ `(H, W, 3)` の配列です。
