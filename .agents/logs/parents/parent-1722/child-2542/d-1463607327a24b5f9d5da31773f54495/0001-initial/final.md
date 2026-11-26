`video_000/clip_064` の採点用ラベルを4視点の元動画から作成しました。全16件です。

| 種別 | certain | probable | uncertain |
|---|---:|---:|---:|
| 打球 | 7 | 1 | 0 |
| バウンド | 7 | 0 | 0 |
| ネット接触 | 1 | 0 | 0 |

f400の打球は肩で接触が隠れたため `probable` としました。f1184の白帯接触後、打球側への最初の着地 f1202 をポイントの終端に含め、次のバウンド f1243 をポイント外としました。視点間で接触が最大約4フレームずれて見えた箇所は、その差を `frame_uncertainty` に含めています。

`unlabelable` は f442–575 と f882–981 です。どちらも高いロブが4視点の画角から外れた区間で、境界は目視による近似です。

4本の動画を固定 manifest と照合し、全1,252フレームの時刻と根拠画像16枚の採用フレームを確認しました。 [labels.json](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_064/labels.json) · [根拠画像](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_064/evidence)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}