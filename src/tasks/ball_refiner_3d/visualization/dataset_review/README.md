# 3D Refiner Dataset Review

```bash
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.review_dataset
```

既定は `http://127.0.0.1:8787`。
3D GT・三角測量入力・モデル予測を重ね合わせ、または横に並べる。
座標・距離誤差・速度・欠損timelineとイベント確率を、同じframeに同期する。
イベントグラフはGaussian教師とsoftmax確率を比較する。
2D表示・camera選択・2Dグラフは廃止。3D上のcamera配置の表示は残す。

物理head付きのモデルでは、次も表示する。
- 積分軌道2本：予測イベントで区切った区間、GTの区間で積分したもの（上限）。
- 予測とGTの区間境界のtimeline。
- 場（風・k_drag・k_magnus）とsurfaceの予測とGT。

線形補間も比較系列として選べる。
ラリーごとの表は、座標RMSEに加えて、GT飛行区間内の差分による速度・加速度・jerk比・非物理加速度率を示す（`physics_eval.v1` と同じ定義）。
力モデルの当てはめ残差はCPUで1系列20秒ほどかかるため、ラリー単位では計算しない。
「学習時のtest評価」の表は、各runの `predictions(_last)/metrics.json` を並べる。checkpointとdatasetのhashは `diagnostic_metrics.json` で照合する。

- `checkpoints.py`: 重み形式・dataset・FPSを照合して候補とvalidation最良の★を表示。run名と物理headの有無をラベルに出し、保存済みtest評価を対応付ける。
- `contracts.py`: リクエストの検証。既存実験の評価条件の読込は `configuration/evaluation.py`。
- `service.py`: 共通データの拡張、公開Predictorによる推論、保存済み評価の照合。
- `payload.py`: 3D表示用データと指標のみをブラウザへ送る。
- `artifacts.py` / `prepare.py`: checkpoint・dataset・条件・予測hashを対応付けたcache。
- `web.py` / `static/`: HTTP境界と画面操作。

拡張条件・seedを変えると以前の予測を無効化する。ノイズは三角測量前の検出座標に適用する。
CPUは画面から再推論し、CUDA要求は共有training queueへ送る。
`--prepare-saved-predictions` は対応best checkpointのtest予測をCPUで再計算し、別review cacheへ保存する。
cacheの形式は `ball_refiner_3d.review_predictions.v2`（物理headの配列を含む）。形式はcacheの保存先hashにも含めるので、v1のcacheは参照されない。
学習済み重みや元の学習結果は上書きしない。

設定JSONは `ball_refiner_3d.event_review.v2`。入力hashと重みhashを照合して再現する。
camera選択を持つ旧v1画面設定は明示的に拒否する。
[task README](../../README.md)が全体の起点。
