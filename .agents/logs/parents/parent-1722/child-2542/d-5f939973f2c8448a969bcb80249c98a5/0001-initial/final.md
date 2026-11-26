調整用 `video_000/clip_001` の全561フレームを4視点の元映像だけで確認し、11件を記録しました。全件がポイント内です。

| 種別 | certain | probable | uncertain |
|---|---:|---:|---:|
| 打球 | 6 | 0 | 0 |
| バウンド | 4 | 1 | 0 |
| ネット接触 | 0 | 0 | 0 |

判定不能区間は0件です。フレーム328のバウンドは日陰で接地の瞬間が見えにくく、`probable`（±1フレーム）としました。フレーム494の頭上の返球は、振り下ろしが小さくラケット面で合わせていたため `volley` としました。映像はフレーム525のバウンド後、球が飛行中に終了します。

[labels.json](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_001/labels.json) と [根拠画像11枚](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_001/evidence/) を保存しました。JSONのフレーム数・FPSがmanifestと一致し、全イベントに根拠画像が対応することを確認しました。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}