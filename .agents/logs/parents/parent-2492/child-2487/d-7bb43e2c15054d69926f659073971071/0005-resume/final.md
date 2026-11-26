RT-DETRv4-L/X の計測を PT16・TRT16 に絞り、精度・速度の結果に `PT32未計測` を明示するよう変更しました。TensorRT の ONNX 書き出しとエンジン構築は実行手順に含めています。

81件のテストと pre-commit が通過しました。`codex/parent-2492-issue-2487-rtdetrv4` の `88ec50e5f777cbecb1dfecf54f8c6353a60612f6` を push 済みです。[CI 実行](https://github.com/Motoki0705/tennis-lab/actions/runs/45543633152)

L/X の再計測を共有 GPU キューに登録しました。精度は固定6,844画像、速度・VRAMは固定96画像のバッチ1・2で測ります。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204420843572522223_2026-11-25T03-59-16-427477776Z_81109_i2487-rtdetrv4-l-suite-fp16-v3.job","0-8204420843246769410_2026-11-25T03-59-16-753230589Z_81238_i2487-rtdetrv4-x-suite-fp16-v3.job"]}}