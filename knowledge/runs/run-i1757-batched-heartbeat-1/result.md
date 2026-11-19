# meiji_4cam 3クリップ 条件3の実測

`batched-heartbeat-1` は `fb6255a2853a6992e3d96287d7b989b681c2e336` の一括CLIで、clip_000／027／037を3回ずつ生成した。4カメラ、各回3,532 camera-frame。2026-11-19 22:32 JSTにqueueへ登録し、2026-11-20 00:14 JSTに本計測が終了した。共有queueは `done`、本計測は `complete`、終了コード0。公表済みの設定と許容差を使用した。 [再現手順](../run-i1757-video-batch-meiji/reproduce.md) [固定許容差](../run-i1757-video-batch-meiji/tolerances.json)

| repeat | 計測区間／worker起動込み（秒） | clips/hour | camera-frame/s | allocated／reserved peak（GB） | process RSS peak（GB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| first | 1967.61 / 1984.33 | 5.44 | 1.780 | 3.985 / 4.452 | 9.273 |
| warm 1 | 1859.53 | 5.81 | 1.899 | 3.985 / 4.452 | 9.575 |
| warm 2 | 1890.44 | 5.71 | 1.868 | 3.985 / 4.452 | 9.583 |

warm中央値1874.99秒、標本標準偏差21.86秒（n=2）。条件2の1802.76秒に対して+72.23秒（+4.01%）。条件3の最短warmも条件2の最長warmより51.13秒長かった。旧逐次の2018.28秒に対しては7.10%短い。各repeatでGPU管理対象7モデルを各1回ロード・解放し、CPU AFLinkは12回ロードした。recorderのエラーは全回0件。 [条件2の実測](../run-i1757-video-batch-meiji/model-reuse-comparison.md) [全計測項目](batched-heartbeat-1.json)

preflightの空きはRTX 5060 TiでVRAM 15.89 GB、RAM 37.89 GB、disk 287.98 GB（10進単位）。PyTorch allocatorの上限は10 GiB。dispatchの作業量ピークは2,344,016,548 bytes（2.18 GiB）で、設定上限3 GiB内だった。共有store cacheの予算は0 byte。diskの実read/writeは first 1.71/2.25 GB、warm 1 2.51/2.20 GB、warm 2 0.68/2.20 GB。OSのcache状態は通常運用のままで、論理I/OはJSONに保存した。中間配列保存／scene export／公開markerのinclusive span合計（秒）は11.89/14.26/3.57、18.53/13.93/3.59、17.29/13.92/3.57。

## 条件2との差を占めた区間

C1 recorderの `stage_wall_seconds` は重複を除くwall区分。各セルはfirst／warm 1／warm 2の順で、増減はwarm 2回の平均差。条件2は `795d63df`、条件3は `fb6255a2` のため、段の記録境界と実装の違いを含む。

| recorder stage | 条件2（秒） | 条件3（秒） | warm増減（秒） |
| --- | ---: | ---: | ---: |
| artifact_load | 11.70 / 11.83 / 12.02 | 12.26 / 12.24 / 12.00 | +0.20 |
| batch_dispatch | — | 42.76 / 53.16 / 62.72 | +57.94 |
| decode | 22.46 / 25.34 / 25.87 | 25.64 / 24.62 / 24.63 | −0.97 |
| inference | 1194.43 / 1163.53 / 1170.48 | 1201.56 / 1144.34 / 1157.04 | −16.31 |
| measurement | 85.48 / 42.74 / 42.94 | 86.88 / 43.61 / 43.79 | +0.86 |
| model_load | 66.76 / 66.43 / 64.01 | 110.33 / 104.23 / 109.16 | +41.48 |
| model_unload | 7.07 / 7.02 / 7.39 | 7.61 / 7.08 / 7.26 | −0.03 |
| postprocess | 236.53 / 235.94 / 241.43 | 178.72 / 174.16 / 179.33 | −61.94 |
| preprocess | 172.81 / 174.86 / 178.50 | 183.30 / 163.10 / 157.44 | −16.41 |
| publication | 19.76 / 20.19 / 19.49 | 20.44 / 22.74 / 31.72 | +7.39 |
| save | 11.69 / 14.37 / 11.52 | 11.89 / 18.53 / 17.29 | +4.97 |
| transfer | 11.64 / 11.87 / 11.50 | 12.58 / 12.78 / 13.17 | +1.29 |
| unattributed | 19.66 / 20.31 / 20.64 | 68.84 / 74.46 / 70.54 | +52.02 |
| video_read | — | 1.68 / 1.59 / 1.60 | +1.59 |

増分が大きいのはdispatchの子span外に帰属する区間、帰属先未計測の区間、モデル読込。モデルをロードした回数は同じで、読込増分は特にappearance（warm平均3.11→24.87秒）、HMR2（21.39→30.67秒）、ViTPose（20.54→26.64秒）、GVHMR（2.41→7.25秒）にあった。読込時間が増えた要因は、この収録から特定できない。

inclusive forwardではDINOが3,532→1,767回になり、warm平均566.26→527.98秒だった。ViTPoseは1,696→6,770回で453.20→455.22秒、HMR2は120→439回で43.22→45.77秒。条件2も各camera内でネイティブのcrop batch（ViTPose 16、HMR2 16）を使うため、条件3では上限の異なる横断batchを多数実行した。外観CLIPのforwardは約70秒で同程度。人物対応CLIPは両条件16回のままwarm平均2.57→20.56秒で、記録上はこのforwardも増分に寄与した。Ballは両条件とも216回で約22秒。

`batch_dispatch + preprocess + postprocess` はwarm平均415.37→394.95秒。dispatchの+57.94秒には、旧経路でpreprocess・postprocessへ入った作業が移った影響があり得る。dispatchの全時間を追加費用とみなす根拠はない。帰属先未計測の+52.02秒を個別のスケジューラや保存処理に結び付けるspanもない。条件2と同じHEAD上の反復比較は未実施。

## 実際のforward shapeと充填率

3 repeatで同じ分布を得た。括弧内はforward回数。Courtは `video_shared` の成果物をロードしたため、Court推論のdispatchは0。CLIPの人物対応は外観特徴とは別経路で、右欄は外観特徴のdispatchの値。

| モデル | 実際の `batch_shapes` | 横断dispatch件数 | 充填率 | 複数clipを含む件数 |
| --- | --- | ---: | ---: | ---: |
| Ball | [1,32,3,720,1280] (216) | 216 | 100% | 0 |
| DINO | [2,3,750,1333] (1765)、[1,3,750,1333] (2) | 1767 | 99.94% | 1524 |
| ViTPose | [8,3,256,192] (6768)、[6,3,256,192] (2) | 6770 | 99.99% | 6467 |
| CLIP-ReID | [16,3,256,128] (1695)、[7,3,256,128] (2)、[8,3,256,128] (1)、[64,3,256,128] (11)、[48,3,256,128] (1) | 1694 | 99.90% | 1617 |
| HMR2 | [4,3,256,256] (439) | 439 | 100% | 407 |

ViTPoseのflipは各cropを2枚にする。CLIPの16件shapeには人物対応側の4 forwardも含まれる。DINOは単独経路と同じ750×1333、Ballは両条件で同じ32×720×1280で、横断化による入力の空間shape増加は記録されていない。端数も少なく、今回の遅延をbatchの空きや追加paddingで説明する証拠は得られなかった。

## 保存・再読込と固定許容差

3 repeatの全9 sceneを公開markerから読み直して検証した。全件4人、player証拠によるcourt_sideはcam0〜cam3の順で `[false,false,true,true]`、ball参照と一致。clip_000／027／037の有効なボールframeは毎回159／123／128だった。27の123と37の128は各clipの総frame数235／292の一部で、2D未検出や3D棄却を含む。

条件2および旧逐次と各repeatを照合し、全6組のsceneが固定許容差を超えた。条件2とのfirst比較では入力・許容差、公開設定の選択項目、Torch/CUDA/threadsが一致した。最初の差は全clipでBallのheatmap・候補と、DINOの検出box・score・frameごとの採用行に見つかった。ballの選択窓と窓内frame番号は同じだった。

| clip | 条件2→条件3のボール3D有効frame | 有効性が変わったframe（0始まり） | 共通有効のボール最大成分差（m） | 共通有効の人物root／関節 最大成分差（m） |
| --- | ---: | --- | ---: | ---: |
| 000 | 159→159 | 0件 | 0.49753 | 0.00250 / 0.30056 |
| 027 | 128→123 | 8、71、147、148、149 | 0.06657 | 0.02347 / 0.16474 |
| 037 | 129→128 | 286 | 0.14527 | 0.00248 / 0.42718 |

clip_027のframe 8は3 cameraのBall検出とball_pointsの採用が同じまま、3Dの棄却理由が `REPROJECTION` に変わった。frame 71、147〜149は `EXCESSIVE_SPEED`。frame 148ではcam2の検出採用が両条件とも有効で、ball_pointsの採用が変わった。frame 232は両条件で無効のまま、cam0のball_pointsが外れ、棄却理由が `BEHIND_CAMERA` から `INSUFFICIENT_VIEWS` へ変わった。clip_037のframe 286でもcam0のball_pointsが外れ、3Dが `REPROJECTION` で棄却された。いずれも検出候補・座標の差を含む。

人物対応の `player_ids` はclip_000で1要素、027で14要素変わった。sceneの最終 `player_track_ids` とroot/heading/SMPLの有効maskは同じだった。clip_000では6関節要素の3D有効性と棄却理由が変わった。 [旧逐次との比較](baseline-comparisons.json) [同一HEADでの逐次対照](../run-i1757-serial-fb6255a-first-1/result.md)

jobは `0-8204904851837375583_2026-11-19T13-32-28-162624416Z_2750129_i1757-batched-heartbeat-1`。共有queueのログは `.training_queue/logs/<job-id>.log`。出力は `outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/batched-heartbeat-1/`。各 `attempts/*/measurement.json` の `recording` がC1 JSONを指し、`compare-*.json` にnode、配列、許容差からの逸脱を残した。 [数値とshape](batched-heartbeat-1.json)
