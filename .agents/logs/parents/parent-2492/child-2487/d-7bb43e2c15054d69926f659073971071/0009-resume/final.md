RT-DETRv4 L/X とも ONNX export と `trt16-fp32attn` plan の構築に成功しました。両ジョブは、新 plan の最初の batch 1 ケースで非有限値を検出して終了コード 1 で停止しました。初回の診断コードは生出力の保存前に終了したため、どの出力に非有限値があったかと、残りの3ケースは未確認です。 [初回のログ・plan SHA-256・実行記録](https://github.com/Motoki0705/tennis-lab/blob/0b07efeaff245c5f782df70e5005f7b6cd5b35c5/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4-results.json)

生出力を保持し、画像ごとの NaN・Inf 件数を記録して後続ケースへ進むよう診断を修正しました。SHA-256 を照合した構築済み plan を使い、L/X を各1件再投入しました。全6,844画像の精度・速度計測へ進むかは、この4画像の結果で判断します。

`codex/parent-2492-issue-2487-rtdetrv4` の `0b07efeaff245c5f782df70e5005f7b6cd5b35c5` を push しました。pytest 90件（追加3件、13.12秒）、pre-commit、knowledge 検証が通過し、CI を起動しました。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45635969206)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204382571766266034_2026-11-25T14-37-08-233733965Z_1369773_i2487-rtdetrv4-l-trt16-fp32attn-v2.job","0-8204382571503597413_2026-11-25T14-37-08-496402586Z_1369831_i2487-rtdetrv4-x-trt16-fp32attn-v2.job"]}}