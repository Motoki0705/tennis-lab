`video_000/clip_009`（tuning）の全974フレームを4視点の元動画で確認し、[labels.json](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_009/labels.json) と [根拠画像21枚](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_009/evidence/) を保存しました。4視点のフレーム時刻と、JSON・根拠画像の対応を検証しました。

| 種別 | certain | probable | uncertain | 計 |
|---|---:|---:|---:|---:|
| 打球 | 10 | 1 | 0 | 11 |
| バウンド | 6 | 3 | 0 | 9 |
| ネット接触 | 1 | 0 | 0 | 1 |
| **計** | **17** | **4** | **0** | **21** |

判定不能区間は0件です。第1サーブはフレーム105で白帯に接触するレットでした。フレーム131・154の接地、368の回収時のラケット接触、967のコート外バウンドは、接触の瞬間が不鮮明なため `probable` としました。904〜907では、cam2で球がラケットの上を間隔を保って通過することを確認しました。

第1サーブはレットの最初の着地まで、第2サーブはラリー最後のインの着地までを `in_point` とし、計12件です。手で飛球を受け止めたフレーム225・400は、身体との接触としてポイント外の `shot/other` に含めました。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}