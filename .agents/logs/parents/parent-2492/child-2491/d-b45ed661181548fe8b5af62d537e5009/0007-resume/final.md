**TRT16は今回の試行では実行不能です。** `torch.export` はcuDNNのTF32設定、TorchScriptはLightningの`Trainer`参照で停止し、ONNX・エンジンとも生成されませんでした。書き出しを再試行する場合の修正候補と、既存8フレーム試行の関節差P95 45.12 mmも記録しました。[記録PR #2534](https://github.com/Motoki0705/tennis-lab/pull/2534)

保存済みGVHMRとFastHMR PT16の12 NPZを照合し、代表6 clipの1,711時刻と両方式各6,617人・フレームの対応を確認しました。`clip_003` の試作で4種類の同期動画を確認し、全6 clipのCPU描画を起動しました。[試作版](https://github.com/Motoki0705/tennis-lab/blob/8a25fe12b543656cb1e883e7251bd1a3fd299fd0/outputs/fasthmr-setup/review-pt16/preview/index.html) · [全6 clipの出力先](https://github.com/Motoki0705/tennis-lab/blob/8a25fe12b543656cb1e883e7251bd1a3fd299fd0/outputs/tennis_scene/evaluate/fasthmr-qualitative/pt16-20261125)

#2497 `13e508cb7` を取り込み、45件のテストとpre-commitを通して `8a25fe12b543656cb1e883e7251bd1a3fd299fd0` をpushしました。PT16参考速度の固定入力とB=1/2/4の順序は照合済みです。CPU描画の負荷と計時が重ならないよう、描画完了後にparity未判定モードで速度ジョブを投入します。[実装PR #2501](https://github.com/Motoki0705/tennis-lab/pull/2501)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261125T070233Z_387170703f99_fasthmr-render-review-6clips"]}}