#2304 の終了済みジョブを照合し、修正後の4件を共有キューへ登録しました。前回の待機スクリプトは削除しました。

- **MSDAの3方式比較**。GridSampleとプラグインはエンジンのシリアライズまで成功し、実行層の照合で停止していました。TensorRTの表示上の層種別から、ONNX演算名を含む `Metadata` での照合に修正しました。部分TRT化は初回から計測できています。
- **TensorRT FP16**。位置埋め込みの `Cast` がFP32を出力したため、FP16との `Add` をパーサーが拒否しました。変換器を変更し、ビルド前にONNXの型検査を追加しました。
- **Nsight Compute**。15分の上限で打ち切られました。FP32は1,100／1,862件、FP16は1,052／2,370件だったため、部分集計は採用せず、同じkernel replayで上限を45分にしました。
- **範囲外shape**。`749×1333` を推論前に終了値1で拒否しました。 [試行記録](https://github.com/Motoki0705/tennis-lab/blob/523c4a9ad42c2d130d335ea2a67fdb9e0144dfc6/knowledge/runs/run-i2304-dino-tensorrt/attempts.json)

登録したジョブは `i2304-msda-probe-v2`、`i2304-dino-prepare-build-grid-2-v2`、`i2304-dino-ncu-0-v2`、`i2304-dino-ncu-1-v2` です。

再開時は3方式の速度と誤差、FP16エンジンの実行精度、両プロファイラのkernel件数を照合します。その結果から採用方式を決め、残るFP16計測とINT8・FP8のPTQへ進みます。CPU検証は86件通過しました。ブランチは `codex/trt-dino` で、新規コミットとPRは未作成です。