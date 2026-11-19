# meiji_4cam 3クリップの逐次計測

実行commitは `8c91bf074816b3d809ac2548f8d8050fbcda3ac9`。共有training queueのall枠でprobeを通過し、初回1回と同じworker内のwarm 2回で、各回の保存先を空にして3クリップすべてを生成した。対象は `video_000` の clip_000・clip_027・clip_037、計3,532 camera-frame。RTX 5060 Tiで実行し、probeと本計測はいずれも終了コード0だった。queue jobと固定入力・重みの参照、各回の実測値は[計測記録](serial-heartbeat-2.json)にある。

`serial-heartbeat-1` の終了コード143は、親の実行記録によると起動元の共通ランナーによるqueue workerの回収だった。下表は親が独立して起動したqueue workerで完走した `serial-heartbeat-2` の結果。

| repeat | 計測区間（秒） | worker起動を含むwall（秒） | 完了clips/hour | camera-frame/s |
| --- | ---: | ---: | ---: | ---: |
| 1 / first | 2118.01 | 2131.75 | 5.07 | 1.66 |
| 2 / warm | 2031.50 | 2031.50 | 5.32 | 1.74 |
| 3 / warm | 2005.06 | 2005.06 | 5.39 | 1.76 |

初回のworker importは13.75秒。warmの中央値は2018.28秒、範囲は2005.06〜2031.50秒、標本標準偏差は18.69秒（n=2）。スループットの分母にはworker起動を含むwallを使った。

## 段階別時間と資源

C1 recorderの `stage_wall_seconds` を秒単位で示す。各spanの重複を除いたwall区分で、infer・postprocessの定義は実行commitのコードによる。

| stage | first | warm 1 | warm 2 |
| --- | ---: | ---: | ---: |
| inference | 1206.29 | 1172.75 | 1169.71 |
| postprocess | 325.10 | 328.00 | 318.28 |
| model_load | 211.22 | 192.10 | 181.45 |
| preprocess | 164.39 | 164.43 | 165.90 |
| measurement | 88.35 | 45.56 | 44.61 |
| save | 24.27 | 26.55 | 26.47 |
| decode | 23.97 | 26.76 | 24.76 |
| model_unload | 20.30 | 20.82 | 20.66 |
| transfer | 11.69 | 11.71 | 11.67 |
| artifact_load | 10.77 | 10.86 | 10.23 |
| publication | 6.65 | 6.64 | 6.58 |
| unattributed | 23.80 | 24.42 | 24.01 |

中間配列の保存は11.18／13.36／13.39秒、scene exportは14.48／14.59／14.46秒、dataset markerの公開は3.56／3.58／3.56秒だった。これは該当spanのinclusive集計で、上のwall分解と計測範囲が異なる。

PyTorchのallocated peakは全回4.185 GB、reserved peakは全回4.461 GB。process-lifetime RSSは初回9.547 GB、warmは両回9.647 GBだった。diskからの `read_bytes` は385.3／835.3／1.0 MB、`write_bytes` は2224.6／2177.8／2177.8 MB（10進単位）。OSのcache状態は通常運用のため、論理read/writeのカウンタもJSONに残した。recorderの計測エラーは3回とも0件。

ball、DINO、ViTPose、appearance、HMR2、GVHMR、body geometryは各repeatでそれぞれ3回ロード、3回解放した。CPU AFLinkは各回12回ロードを記録した。3回のbatch分布と充填率はJSONに保存した。初回はDINO 3,532画像とBall 216窓がすべてB=1。ViTPoseは99.75%、人物外観CLIPは98.58%、人物対応CLIPは79.69%、HMR2は91.46%だった。

## 固定許容誤差による出力確認

3回の全3組を[事前固定した許容誤差](tolerances.json)で比較した。入力、重み、設定、runtimeのpinが一致し、各組で47 node × 3クリップの保存payloadとscene配列がexactだった。ID、採否、mask、停止理由にも差がなかった。9件すべて公開マーカーからsceneを再読込して検証した。

player証拠のcourt_sideは全9件でcam0〜cam3の順に `[false, false, true, true]` となり、ball参照と一致した。全sceneで人物は4人、clip_000／027／037の有効なボールフレームは各repeatで159／122／130だった。

raw JSON、全比較結果、sceneは `outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/serial-heartbeat-2/` にあり、recorderは `attempts/{1-first,2-warm,3-warm}/recording.json`、matrixは `compare-*-to-*.json`。本計測の終了は `measurement/terminal.json` と共有queueの `done` が一致した。

条件2とCLIP再利用条件は、#2227・#2271を含む別commitで計測する。#2227ではX3Dのpostprocessの記録境界も更新されたため、条件間のstage差にはcommitの差が含まれる。条件間の速度・品質、失敗scene、再開、cache混在の確認は残っている。
