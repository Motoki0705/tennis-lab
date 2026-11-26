`video_000/clip_027` の235フレームを4視点の元映像で確認し、採点用イベントを4件記録しました。

| 種別 | フレーム | status |
|---|---:|---|
| サーブ | 62 ±3 | `certain` |
| サービスのバウンド | 108 ±3 | `certain` |
| フォアハンドのリターン | 149 ±3 | `certain` |
| ベースライン外のバウンド | 215 ±1 | `probable` |

打球2件、バウンド2件、ネット接触0件、`unlabelable` 0区間です。同じ接触が視点間で最大約3フレームずれて見えたため、最も明瞭な視点の番号を採用し、差を不確かさに含めました。215フレーム付近は日陰で接触点が不鮮明でした。前後の反発から判定し、ポイントを終えた着地として `in_point` に含めました。

manifestとのフレーム数・FPSの一致と、根拠画像4枚を確認しました。[labels.json](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_027/labels.json) · [根拠画像](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_027/evidence)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}