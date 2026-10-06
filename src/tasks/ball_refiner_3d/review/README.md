# 3D Refiner Dataset Review

```bash
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.serve_coordinate_review --port 8787
```

`http://127.0.0.1:8787` で共通rallyを選び、cameraごとの2D GT・拡張後観測と、
3D GT・三角測量入力・3D Refinerの推論を同期表示する。
2Dは入力の確認用であり、2Dモデルの選択・推論は提供しない。
イベント確率グラフはGaussian教師とモデルのsoftmax確率を重ね、欠損・frame送りと同期する。

- `outputs/ball_refiner_3d` と `ckpt/ball_refiner_3d` の対応checkpointを候補表示する。
- 同じ評価条件のvalidation RMSE最良に★を付ける。異なるデータ・FPS・旧schemaは非対応理由を表示する。
- 欠損選択率・左右幅・ノイズ・seedの変更を可視化し、変更後は以前の予測を無効化する。
- CPUは画面から再推論できる。CUDAは共有training queueを経由する。
- 保存済み評価はcheckpoint・dataset・予測のhashと入力条件を照合する。読み替えない。
- `--prepare-saved-predictions` で対応best checkpointのtest予測をCPU再計算し、別review cacheに保存できる。

旧checkpointにはイベントヘッドがないので候補には採用しない。推論精度の評価には新形式での学習が必要。
学習・損失・データ契約は[task README](../README.md)を参照。
