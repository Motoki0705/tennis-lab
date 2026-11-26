`video_000/clip_040` の全417フレームを元動画4視点で確認し、採点用ラベルと根拠画像9枚を保存しました。 [labels.json](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_040/labels.json) · [根拠画像](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_040/evidence)

| 種別 | certain | probable | uncertain |
| --- | ---: | ---: | ---: |
| 打球 | 3 | 1 | 0 |
| バウンド | 4 | 0 | 0 |
| ネット接触候補 | 0 | 0 | 1 |

`unlabelable` は0区間です。f198の打球は打点が遮られたため前後の球の動きで判定しました。f246は球と白帯が重なり、物理接触の有無は未確定です。

視点間で同じ接触が約1〜2フレームずれて見えるため、ラベルはcam0の時刻にそろえました。ポイントを終えたf403の第2バウンドも `in_point=true` に含めています。JSONのフレーム数・FPSをmanifestと照合し、全9件の根拠画像を確認しました。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}