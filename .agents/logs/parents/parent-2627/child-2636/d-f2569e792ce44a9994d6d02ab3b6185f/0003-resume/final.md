保存設定が CUDA を要求していた問題を修正し、TensorRT を初期化せず保存済み特徴を使う CPU 再生を再起動しました。

停止14 clip の論文用PDFを作成しました。全14図を読み出せることと、再生成したPDFがバイト単位で一致することを確認しました。画像・索引・観測記録のcommitは `618890b83dcb3362171d2b4596ba8a65e59ee5a2` です。再生結果を受けてclip別の判定を確定し、記録専用PRにまとめます。 [図の索引](https://github.com/Motoki0705/tennis-lab/blob/618890b83dcb3362171d2b4596ba8a65e59ee5a2/knowledge/runs/run-meiji-stopped-clip-observations/figures/index.csv)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T115210Z_4a7282d419c9_meiji-stopped-clip-replay-cpu"]}}