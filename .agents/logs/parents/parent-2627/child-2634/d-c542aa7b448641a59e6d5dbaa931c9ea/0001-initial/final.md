#2634 の全件CPU評価を起動しました。Meiji 3cam 7本と4cam 69本で、選別なし、画像2D、コート滞在によるraw ID選別、現行の断片連結付き選別を比較します。[実験設計と試走記録](https://github.com/Motoki0705/tennis-lab/issues/2634#issuecomment-7190663054)

3cam全7本の入力を照合しました。4cam `clip_027` の試走では、現行条件の選別マスク・元トラック行・3D関節有効マスクが保存値と一致しました。pytestは34件通過し、run 10・11の13条件を[出典付きCSV](https://github.com/Motoki0705/tennis-lab/blob/dfb03c0785c7a7a41850c5cd6909b1bb383a0b75/knowledge/runs/run-i2634-player-selection-ablation/historical-2d.csv)にまとめました。

実行HEADは `dfb03c0785c7a7a41850c5cd6909b1bb383a0b75` です。全件結果の受領後に、指標の集計、主張の支持判定、記録専用PRの作成を進めます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T114001Z_626be073a61f_i2634-player-selection"]}}