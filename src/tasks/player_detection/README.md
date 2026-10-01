# Player Detection

出力先は [タスク出力規約](../OUTPUTS.md) に従います。

chat-annotation の処理済み player ラベルから、テニスプレーヤー（主コート上の選手）の検出器を作ります。
公式 DINO 4-scale Swin-L の COCO person logit（class id 1）を「tennis player」として fine-tune し、
`DinoPersonDetector` がそのまま読める DINO 形式の checkpoint を export します。
pipeline 側は `people_models.dino_checkpoint` を差し替えるだけで使えます。

## 流れ

1. `scripts.generate_dataset`: `outputs/chat_annotation` の processed player ラベルから
   `data/player_detection/<version>/` の frame store を作ります。player を含む frame だけを保存し、
   clip-local の track ID と時刻を残します（将来の ID tracking 用）。
   split の単位は YouTube の source 動画です。同じ version への上書きは拒否します。
   保存形式と読み出し API の正本は [data/store.py](data/store.py) です。
2. `scripts.preview_dataset`: split ごとの frame と box を重ねた preview を書きます。
3. `scripts.train`: Lightning で fine-tune します。frame の選別（未 review、`unresolved` player を含む frame、
   画面外にほぼ出た box の除外）は [data/detection_dataset.py](data/detection_dataset.py) が所有し、
   除外件数を split ごとに表示します。
4. `scripts.export_checkpoint`: Lightning checkpoint を DINO 形式の `.pth` に変換します。
   変換前に上流 DINO へ strict load して構造を確かめ、出自を `tennis_lab` キーに記録します。
5. `scripts.evaluate`: COCO 版と export 版など複数の checkpoint を、pipeline と同じ前処理で比較します。

設定の正本は [configs/](configs/) です。評価の `input_size` と `score_threshold` は、
pipeline の `people_models.runtime.dino_detector` と一致させる必要があります。
DINO の denoising は CUDA 前提のため、学習と評価は GPU（training queue 経由）で実行します。

```bash
.venv/bin/python -m src.tasks.player_detection.scripts.generate_dataset
.venv/bin/python -m src.tasks.player_detection.scripts.train
.venv/bin/python -m src.tasks.player_detection.scripts.export_checkpoint \
  export.lightning_checkpoint=<output-root 相対 .ckpt> \
  export.destination=player_detection/<name>.pth
.venv/bin/python -m src.tasks.player_detection.scripts.evaluate \
  '+evaluate.checkpoints.player_ft=player_detection/<name>.pth'
```

DINO の構築と前処理は [src/submodules/models/dino/architecture.py](../../submodules/models/dino/architecture.py) を
推論（`DinoPersonDetector`）と共有します。

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
