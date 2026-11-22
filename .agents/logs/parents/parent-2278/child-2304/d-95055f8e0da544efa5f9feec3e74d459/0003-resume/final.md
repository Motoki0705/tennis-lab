#2304 は、全体計測に標準GridSampleを採用します。小規模試行では3方式とも動作し、中央値はGridSample 0.496 ms、TensorRTプラグイン 0.091 ms、部分TRT化 0.201 msでした。GridSampleを選んだ理由は、CPU校正とTensorRT実行で同じONNXグラフを使えることです。 [方式比較](https://github.com/Motoki0705/tennis-lab/blob/74abdc4d6d536e443e524008bd651470d6871137/knowledge/runs/run-i2304-dino-tensorrt/README.md%23msda%E3%81%AE%E6%96%B9%E5%BC%8F%E6%AF%94%E8%BC%83)

PyTorch FP32／FP16は全指標を取得し、nsysとncuのkernel件数がそれぞれ1,862件[local path],370件で一致しました。TensorRT FP16の前回試行はONNXノードの依存順で停止しました。順序を修正し、保存済みグラフが厳密な型検査を通ることを確認しました。CPUテストは86件通過しました。 [計測結果](https://github.com/Motoki0705/tennis-lab/blob/74abdc4d6d536e443e524008bd651470d6871137/knowledge/runs/run-i2304-dino-tensorrt/metrics.json)・[試行記録](https://github.com/Motoki0705/tennis-lab/blob/74abdc4d6d536e443e524008bd651470d6871137/knowledge/runs/run-i2304-dino-tensorrt/attempts.json)

次の3件を登録しました。各ジョブは変換・実行精度の検査を通過した場合に本計測へ進みます。INT8・FP8は #2297 のModelOpt設定を使います。

- `i2304-dino-grid-2-v3`
- `i2304-dino-grid-3a-v3`
- `i2304-dino-grid-3b-v3`

再開時に3条件の終了段階と実行精度を照合し、取得済み指標と失敗の根拠をknowledgeへ統合します。ブランチは `codex/trt-dino` で、新規コミットとPRは未作成です。