`video_000/clip_056` の全763フレームを4視点の元映像で確認し、採点用ラベル13件と根拠画像13枚を保存しました。ポイント内は7件、ポイント外は6件です。判定不能区間はありません。

| 種別 | certain | probable | uncertain |
|---|---:|---:|---:|
| 打球 | 6 | 1 | 0 |
| バウンド | 3 | 2 | 0 |
| ネット接触 | 1 | 0 | 0 |

白帯への接触は f681（±3フレーム）です。f270 と f579 は影や遮蔽で接地の瞬間が見えにくく、`probable` としました。f323 の手による捕球は身体接触として `shot/other` に含めました。冒頭のサーブ後は選手が構えを解き、同じ選手がサーブをやり直したため、f223〜323 の球処理をポイント外と判定しました。

指定manifestと4本の動画のフレーム数・FPS、ラベル形式、根拠画像との対応を照合しました。保存先は [labels.json](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_056/labels.json) です。根拠画像は同じディレクトリの `evidence/` にあります。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}