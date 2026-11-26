`video_000/clip_043` の元動画4視点から、採点用イベント16件をラベル付けしました。固定マニフェストと4動画のフレーム数・FPSが一致し、全ラベルに根拠画像が対応することを確認しました。

| 種別 | `certain` | `probable` | `uncertain` |
|---|---:|---:|---:|
| 打球 | 5 | 1 | 0 |
| バウンド | 7 | 2 | 0 |
| ネット接触 | 1 | 0 | 0 |

- **判定不能は f274〜278。** 1本目の球が回収時に選手の手元に隠れました。
- **`probable` は f173・f240 のバウンドと f685 のラケット接触。** f554付近は返球がラケットの上を通過する様子を確認しました。
- `in_point` には、1本目のネットフォルト f119 と2本目のアウト着地 f639 をそれぞれ終端として含めました。

[labels.json](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_043/labels.json) · [根拠画像17枚](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_043/evidence)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}