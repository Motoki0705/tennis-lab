RT-DETRv4 L/X の `trt16-fp32attn` は、両モデルとも比較対象外としました。固定4画像を batch 1・2 と2通りの入力順で実行し、全8ケースで画像ごとの logits 24,000値と BBox 座標1,200値が、後処理前にすべて NaN になることを保存済み配列で確認しました。 [診断記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7166223234)

両ジョブは診断データを保存した後、条件不成立として終了コード1を返しました。最初に NaN が発生する演算は未特定です。全6,844画像の精度と速度は未計測です。 [job ID・plan SHA-256・生出力](https://github.com/Motoki0705/tennis-lab/blob/11631d3be0e6111f2fbad05f08760376ce1d0b34/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4-results.json)

`codex/parent-2492-issue-2487-rtdetrv4` の `11631d3be0e6111f2fbad05f08760376ce1d0b34` を push しました。pytest 90件（7.61秒）、pre-commit、knowledge 検証が通過しました。CI は起動済みで、結果は未確認です。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45640704155)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}