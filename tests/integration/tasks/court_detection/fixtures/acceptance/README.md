# Court後処理の実画像fixture

meiji_4camのframe 0から取得したKP/LINE出力2件。画像のforwardだけを固定し、
hybrid推定、品質判定、crop座標変換、下流maskまで実処理する。GPUやcheckpointを必要としない。
原画像・raw cache・checkpointのSHA-256と領域は各JSONに記録する。

- `supported`: video_000/clip_004/cam0の領域4。線と点の支持がある推定。
- `partial_unstable`: video_000/clip_013/cam3の領域6。境界の2点を除外しても
  solverは成功するが、外挿の不確かさで品質不合格になる。

採取scriptは`knowledge/runs/run-court-acceptance-implementation-20261116/cache_raw.py`。
LINEだけ`round(probability * 65535)`でuint16に量子化し、test内で復元する。
量子化誤差の上限は0.5/65535。KP座標・score・validはfloat32/boolのraw値を保存する。
母集団の精度・受入結果はknowledgeの全件評価を参照し、この2件から一般化しない。
