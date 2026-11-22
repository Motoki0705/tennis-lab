#2304 の3件の失敗原因を確認し、次の投入に向けて修正しました。

- **TRT FP16 GridSample**。ビルド後の検査が、Swinのマスク2入力をQ/K/Vと一緒に数えて停止しました。修正後、保存済みinspectorで主要316演算のFP16と48個のGridSampleの対応を確認しました。全体の推論計測は残っています。
- **INT8／FP8**。INT8は、GPUを隠した状態でModelOptの前処理がTensorRT Builderを作成して停止しました。その前処理が共有ONNXを上書きしたため、FP8はdigest不一致で停止しました。次回は元ONNXを再生成し、条件別コピーで前処理します。校正はORT CPUのままです。 [試行記録](https://github.com/Motoki0705/tennis-lab/blob/53c1c19e873f26c1e8f0b1c59ed22198ed78999f/knowledge/runs/run-i2304-dino-tensorrt/attempts.json)

CPUテストは88件通過しました。指示どおり、次のターンでGridSample版とプラグイン版のFP16全体計測、GridSampleのONNX再生成、INT8・FP8の計5件をまとめて投入できる状態です。投入ターンへの継続指示をお願いします。 [投入スクリプト](https://github.com/Motoki0705/tennis-lab/blob/53c1c19e873f26c1e8f0b1c59ed22198ed78999f/outputs/trt-dino/enqueue-v4.sh)

ブランチは `codex/trt-dino` です。新規コミットとPRは未作成です。

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}