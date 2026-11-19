# モデル再利用とCLIP embedding再利用の比較

meiji_4camのclip_000・027・037（計3,532 camera-frame）を、`795d63df1682d0d86f769d584dfbfa263419ad0d` の一括CLIで各3回生成した。両ジョブが共有queueで `done`、本計測は `complete`、worker終了コードは0、recorderのエラーも全回0件だった。対象は `reuse-heartbeat-1` と `embedding-reuse-heartbeat-1`。逐次基準は同じ入力・重み・精度・許容誤差を使った `8c91bf074` の実測で、コードとpublish identityの一部が異なる。 [逐次基準](serial-heartbeat-2.md)

| 条件 | first 計測区間／起動込み（秒） | warm 1（秒） | warm 2（秒） | warm 中央値／標本標準偏差（秒） |
| --- | ---: | ---: | ---: | ---: |
| モデル再利用 | 1862.82 / 1876.41 | 1797.11 | 1808.41 | 1802.76 / 7.99 |
| CLIP embeddingも再利用 | 1908.02 / 1921.97 | 1823.39 | 1815.32 | 1819.36 / 5.71 |
| 逐次 | 2118.01 / 2131.75 | 2031.50 | 2005.06 | 2018.28 / 18.69 |

逐次に比べてwarm中央値はモデル再利用で10.68%、CLIP再利用で9.86%短かった。品質照合には後述のボール差があるため、速度と品質の両方を満たす受入は未達。起動込みのモデル再利用は各回 5.76／6.01／5.97 clips/hour、1.88／1.97／1.95 camera-frame/s、CLIP再利用は 5.62／5.92／5.95 clips/hour、1.84／1.94／1.95 camera-frame/s。I/O、公開、保存、batch分布とload回数は各runのJSONにある。 [モデル再利用の実測](../run-i1757-model-reuse-heartbeat-1/reuse-heartbeat-1.json) [CLIP再利用の実測](../run-i1757-embedding-reuse-heartbeat-1/embedding-reuse-heartbeat-1.json)

## 段階別時間と資源

C1 recorderの `stage_wall_seconds` を秒単位で示す。各セルは first／warm 1／warm 2 の順。公開・書き込み等のinclusive spanはJSONに残した。

| stage | モデル再利用 | CLIP再利用 |
| --- | ---: | ---: |
| inference | 1194.43 / 1163.53 / 1170.48 | 1205.91 / 1159.49 / 1160.51 |
| postprocess | 236.53 / 235.94 / 241.43 | 256.97 / 250.79 / 255.23 |
| preprocess | 172.81 / 174.86 / 178.50 | 174.80 / 174.06 / 170.83 |
| measurement | 85.48 / 42.74 / 42.94 | 86.48 / 42.63 / 42.92 |
| model_load | 66.76 / 66.43 / 64.01 | 75.97 / 80.75 / 69.96 |
| decode | 22.46 / 25.34 / 25.87 | 22.45 / 26.91 / 23.81 |
| publication | 19.76 / 20.19 / 19.49 | 19.75 / 20.63 / 20.66 |
| save | 11.69 / 14.37 / 11.52 | 10.62 / 13.54 / 16.47 |
| artifact_load | 11.70 / 11.83 / 12.02 | 12.59 / 12.65 / 13.15 |
| transfer | 11.64 / 11.87 / 11.50 | 12.14 / 11.26 / 11.27 |
| model_unload | 7.07 / 7.02 / 7.39 | 7.06 / 6.85 / 6.87 |
| unattributed | 19.66 / 20.31 / 20.64 | 20.44 / 21.18 / 21.07 |

PyTorch allocatorのpeakは全6回でallocated 4.185 GB、reserved 4.461 GB。process-lifetime RSS（GB）はモデル再利用9.208／9.442／9.442、CLIP再利用9.204／9.471／9.484。RSSのprocess-lifetime peakには同じworkerの先行repeatが含まれる。モデル再利用とCLIP再利用とも、C3管理対象7モデルを各repeatで1回ずつロードした。CPU AFLinkは各回12回ロード。

## CLIP再利用の費用

CLIP再利用は全回で816/816件（内訳は000が272、027が288、037が256）の命中だった。人物対応のCLIP forwardは各回16回から0回になり、recorderのinclusive inference spanで2.52〜2.62秒が消えた。人物対応のnode全体は17.59／17.60／17.96秒から17.67／17.40／17.65秒で、追加のartifact読込（0.51〜0.52秒から1.28秒）や画像照合を含めると、総時間の削減は0.31秒以下だった。

外観特徴のnode合計は91.88／91.96／92.20秒から107.90／107.16／107.04秒へ全回で増えた（+16.02／+15.20／+14.84秒）。外観特徴のCLIP forward時間の差は0.11秒以下。この経路は検出のあるframeのデコード済みBGR画像にSHA-256を付ける。毎回生じる外観特徴段の増分と、その処理の追加が対応する。SHA-256単体のspanは取得していない。

warm中央値の差は+16.60秒（+0.92%）で、上記の特徴段の費用と同程度。2回の差は+26.28秒と+6.92秒で、モデル読込や他段のばらつきも乗っている。初回の起動込み差+45.56秒にはmodel_load +9.20秒とinference +11.47秒も含まれる。観測された増加には再利用の前処理費用があり、条件全体の増分の大きさはwarm 2回の範囲に依存する。

## 固定許容誤差による出力確認

3 repeatをそれぞれ同じrepeatの逐次出力と比較した。両条件で全18 sceneを公開マーカーから再読込でき、全件4人・生成成功。player証拠のcourt_sideは全件 `[false, false, true, true]`。入力、チェックポイント、実runtime、許容誤差のpinは一致した。逐次は別commitで、古い外観特徴の保存schemaとpublication設定が異なる。

逐次との比較では、両条件とも全clipでBallのheatmap・候補score・座標が最初の数値差として現れ、ball_points、三角測量、sceneへ伝わった。sceneで連続値の固定許容誤差を超え、`ball_vis` と `ball_rejection_code` も変わった。両条件に同じ変化があり、旧逐次との比較は全組で不合格。

| clip | 3D有効frame 逐次→両条件 | 3D有効性が変わったframe（0始まり） | 共通有効frameの座標成分の最大差（m） |
| --- | ---: | --- | ---: |
| 000 | 159→159 | 0件 | 0.49743 |
| 027 | 122→128 | 8、71、147、148、149、204 | 0.06636 |
| 037 | 130→129 | 193、194、286 | 0.71002 |

同じHEADのモデル再利用とCLIP再利用を直接比較すると、全3組・全3clipで最終scene配列がexactだった。人物IDと対応の離散結果は一致した。外観特徴には照合用のframe SHA、人物対応には命中診断の差があり、clip_000では `diagnostics.association.pairs.31.appearance` も全3組で約2.18×10⁻⁸異なり、固定float64の許容差を超えた。最終ボール出力も両条件で一致した。旧逐次との差がモデル寿命・計画順序・commit差のどれで生じるかは、同一HEADの逐次対照が無く未特定。

raw結果は `outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/` の各runにある。各 `attempts/*/measurement.json` の `recording` がrecorderの場所を指し、`compare-serial-*.json` と `compare-reuse-*.json` に全nodeの差を保存した。

後日の逐次対照では、モデル再利用の初回出力と全node・scene配列が完全一致し、旧逐次との球の差を再現した。 [逐次対照の結果](../run-i1757-serial-fb6255a-first-1/result.md)
