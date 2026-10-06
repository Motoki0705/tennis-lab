# 3D Refiner Dataset Review

```bash
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.review_dataset
```

既定は `http://127.0.0.1:8787`。
3D GT・三角測量入力・モデル予測を重ね合わせ、または横に並べる。
座標・距離誤差・速度・欠損timelineとイベント確率を、同じframeに同期する。
イベントグラフはGaussian教師とsoftmax確率を比較する。
2D表示・camera選択・2Dグラフは廃止。3D上のcamera配置の表示は残す。

- `checkpoints.py`: 重み形式・dataset・FPSを照合して候補とvalidation最良の★を表示。
- `contracts.py`: リクエストの検証。既存実験の評価条件の読込は `configuration/evaluation.py`。
- `service.py`: 共通データの拡張、公開Predictorによる推論、保存済み評価の照合。
- `payload.py`: 3D表示用データと指標のみをブラウザへ送る。
- `artifacts.py` / `prepare.py`: checkpoint・dataset・条件・予測hashを対応付けたcache。
- `web.py` / `static/`: HTTP境界と画面操作。

拡張条件・seedを変えると以前の予測を無効化する。ノイズは三角測量前の検出座標に適用する。
CPUは画面から再推論し、CUDA要求は共有training queueへ送る。
`--prepare-saved-predictions` は対応best checkpointのtest予測をCPUで再計算し、別review cacheへ保存する。
学習済み重みや元の学習結果は上書きしない。

設定JSONは `ball_refiner_3d.event_review.v2`。入力hashと重みhashを照合して再現する。
camera選択を持つ旧v1画面設定は明示的に拒否する。
[task README](../../README.md)が全体の起点。
