`video_000/clip_028` の全1,288フレームを4視点の元映像で確認し、採点用ラベル32件と根拠画像33枚を保存しました。JSONの形式、フレーム範囲、根拠画像との対応を検証しました。

| 種別 | certain | probable | uncertain | 合計 |
|---|---:|---:|---:|---:|
| 打球 | 10 | 2 | 0 | 12 |
| バウンド | 6 | 12 | 1 | 19 |
| ネット接触 | 1 | 0 | 0 | 1 |
| 合計 | 17 | 14 | 1 | 32 |

- f254で、第1サーブ後のポイント外の返球がネットに接触しました。f501は小さな再接地の可能性があり、`uncertain` としました。
- f506–533は、第1球の微小な跳ねと転がりを区別できませんでした。同じ区間の第2サーブ f511 は判定済みです。
- f1178はベースライン際の着地をアウトと読み、続く f1192 の返球をポイント外としました。同じ接触が視点間で最大4コマずれて見えた箇所は、`frame_uncertainty` に反映しました。

[./outputs/tennis_scene/analyze/event-eval-labels/clip_028/labels.json](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_028/labels.json)  
[根拠画像](https://github.com/Motoki0705/tennis-lab/blob/3048c4de1771a9d7b6120ae3148859b478443537/outputs/tennis_scene/analyze/event-eval-labels/clip_028/evidence)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}