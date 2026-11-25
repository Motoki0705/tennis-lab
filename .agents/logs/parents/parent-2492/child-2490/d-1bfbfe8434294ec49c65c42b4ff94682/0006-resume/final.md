RTMPose-M/X の PT16・TRT16 を batch 1・4・8 で計測し、全12条件で `ok` を確認しました。計3,600反復について、固定cropの順序、精度計測と同じ重み・engine、集計値を照合しました。parity は全条件 `PT32未計測` です。

| Batch 1 | Wall P50 / P95 (ms/人) | ピークVRAM (MiB) |
|---|---:|---:|
| M PT16 | 31.04 / 42.98 | 528 |
| X PT16 | 35.76 / 83.82 | 610 |
| M TRT16 | 7.28 / 14.22 | 548 |
| X TRT16 | 9.01 / 12.11 | 698 |

TRT16の中央値は同じ条件のViTPose-H（flipあり）よりMで42.4%、Xで28.7%短く、MのP95は5.6%長くなりました。全batchの実測値、元レポートのパスとSHA-256、再現コマンドを保存しました。 [計測結果](https://github.com/Motoki0705/tennis-lab/blob/218fcef361e2f95a6c86eacefe0a4abbd337b2ca/knowledge/runs/run-meiji-topdown-pose-comparison/rtmpose.md) · [機械可読の実行記録](https://github.com/Motoki0705/tennis-lab/blob/218fcef361e2f95a6c86eacefe0a4abbd337b2ca/knowledge/runs/run-meiji-topdown-pose-comparison/rtmpose_speed.json)

計測HEADは `645bf735dc4a183c46ce1a9448bbd8eee8d0ec30`。記録コミット `218fcef361e2f95a6c86eacefe0a4abbd337b2ca` を `codex/parent-2492-issue-2490-rtmpose` にpushしました。CPUテスト44件、knowledge検査、統合用スクリプトによる12レポートの照合が通過しました。計測HEADのCIは成功済みで、記録HEADのCIは起動済みです。 [計測HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45590029165) · [記録HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45617353160)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}