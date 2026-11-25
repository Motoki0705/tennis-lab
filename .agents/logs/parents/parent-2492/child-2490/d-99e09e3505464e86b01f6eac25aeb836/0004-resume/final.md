Python 3.12 の分離環境で、PyTorch 2.13.0＋CUDA 13.0 と TensorRT 10.14.1.48.post1 を確認しました。初回の1B ONNX出力は163.46秒で完了し、ONNX検査に合格しました。グラフには公式既定値に由来する BF16 の RoPE 座標演算が7箇所ありました。

PT32の定義とバックエンド間の比較条件を揃えるため、1B・5Bとも公式の設定で RoPE を FP32 に固定しました。コミットは `41d23a41a16a6a81941317c5fde6dd08f8df7cb7` です。#2495・#2496 の最新 HEAD を含む状態でテスト47件が通過しました。 [設定と再現手順](https://github.com/Motoki0705/tennis-lab/blob/41d23a41a16a6a81941317c5fde6dd08f8df7cb7/knowledge/runs/run-meiji-topdown-pose-comparison/sapiens2.md)

両モデルの比較用 ONNX 出力と型検査をローカルジョブで開始しました。5B FP32 の CUDA ロード試行、TensorRT ビルド、精度・速度・VRAM計測はこれからです。速度計測には #2496 の固定入力 manifest の公開が必要です。 [#2496 進捗](https://github.com/Motoki0705/tennis-lab/pull/2497)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T133254Z_8a496cc18184_sapiens2-onnx-fp32-rope"]}}