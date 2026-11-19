# 同一HEADでのmeiji_4cam 3クリップ逐次対照

条件3と同じ `fb6255a2853a6992e3d96287d7b989b681c2e336` で、clip_000・027・037の逐次を初回1回だけ実行した。比較目的は球の差を切り分けることで、warmの反復は先行計測にある。元のbenchmarkの各段・設定・重みを使い、[launcher](serial_first_only.py) で実行対象を `attempts/1-first` に限定した。launcherのSHA-256は `907168496157ab1e3bd823658a95b97ea1762ce76ef74bade5440a3d0c0c374c`。 [再現手順](../run-i1757-video-batch-meiji/reproduce.md) [queue投入コマンド](queue-command.txt)

2026-11-20 00:22:17 JSTに共有queueへ登録し、00:28:20にprobe、01:03:55に本計測が終了した。既存のworker PID 1419109のall枠を使用した。本計測の `status: complete` と終了コード0、共有queueの `done` を照合した。heartbeat監視は条件3と共通。probeのRTX 5060 Tiの空きVRAMは15.89 GB、空きRAMは37.52 GB、空きdiskは280.27 GB。

| 条件の初回 | 計測区間（秒） | worker起動込み（秒） | clips/hour | camera-frame/s |
| --- | ---: | ---: | ---: | ---: |
| 旧逐次 `8c91bf074` | 2118.01 | 2131.75 | 5.07 | 1.657 |
| 逐次対照 `fb6255a2` | 2112.18 | 2128.19 | 5.07 | 1.660 |
| モデル再利用 `795d63df` | 1862.82 | 1876.41 | 5.76 | 1.882 |
| 条件3 `fb6255a2` | 1967.61 | 1984.33 | 5.44 | 1.780 |

同一HEADで条件3の初回は6.76%短かった。先行の条件2の初回は逐次対照より11.83%短かった。条件2は別commitで、条件3とのwarm中央値の比較は条件3の実測ノートにある。逐次対照のallocated／reserved peakは4.182／4.461 GB、process-lifetime RSSは9.487 GB、recorderエラーは0件。中間配列保存／scene export／公開markerは10.63／14.11／3.59秒。GPU管理対象7モデルは各3ロード・解放、CPU AFLinkは12ロード。段別時間、I/O、実batch shapeはJSONに保存した。 [逐次対照の計測値](serial-fb6255a-first-1.json) [条件3の時間内訳](../run-i1757-batched-heartbeat-1/result.md)

## 旧逐次と逐次対照で変わった球

固定の入力・チェックポイント・精度・DINO拡張・保存方式・許容差を照合した。3組とも入力・許容差のSHA-256とTorch/CUDA/threadsが一致した。旧逐次と新しい逐次の公開identityにはコード、外観前処理の診断、人物対応設定の明示 `reuse_clip_embeddings: false` の差がある。同一HEADの逐次と条件3は公開identityも一致した。各組47 node×3クリップを照合し、全sceneを公開markerから再読込して検証した。全件4人でplayer証拠のcourt_sideはcam0〜cam3の順に `[false,false,true,true]`。 [固定許容差](../run-i1757-video-batch-meiji/tolerances.json) [比較の数値](comparisons.json)

| clip | 旧逐次→逐次対照の球3D有効frame | 有効性が変わったframe（0始まり） | 共通有効の最大座標成分差（m） |
| --- | ---: | --- | ---: |
| 000 | 159→159 | 0件 | 0.49743 |
| 027 | 122→128 | 8、71、147、148、149、204 | 0.06636 |
| 037 | 130→129 | 193、194、286 | 0.71002 |

旧逐次との9 frameの差を、モデルをクリップごとに解放する新しい逐次でも再現した。新しい逐次と条件2の初回は全141 nodeと全scene配列が完全一致し、固定許容差の品質判定も合格した。この9 frameの差は計測commitを変えた逐次同士に現れ、モデル寿命の変更を必要としない。

最初の数値差はBallのheatmapと候補で、選択した窓と窓内時刻は同じだった。clip_027の148は両条件でcam2のraw検出を採用し、その後のball_pointsの採否が変わって3Dが `EXCESSIVE_SPEED` から有効になった。clip_037の193・194は3Dが `EXCESSIVE_SPEED` で棄却され、286はcam0のball_pointsが有効になり `REPROJECTION` が解消した。clip_000の95・96と027の232は棄却理由が変わり、無効のままだった。旧逐次→新逐次でsceneの人物配列、最終人物ID・maskは完全一致した。commit間にはX3D recorderの同期境界とBallの窓出力保存の変更があり、数値差を生じさせた個別変更やGPU演算は特定できていない。

## 同一HEADの逐次対照と条件3

| clip | 逐次対照→条件3の球3D有効frame | 有効性が変わったframe（0始まり） | 共通有効の球最大成分差（m） | 共通有効の人物root／関節最大成分差（m） |
| --- | ---: | --- | ---: | ---: |
| 000 | 159→159 | 0件 | 0.49753 | 0.00250 / 0.30056 |
| 027 | 128→123 | 8、71、147、148、149 | 0.06657 | 0.02347 / 0.16474 |
| 037 | 129→128 | 286 | 0.14527 | 0.00248 / 0.42718 |

同一HEADでもBallのheatmapと候補に差が出た。Ballは両条件で216 forward、shape `[1,32,3,720,1280]`、同じ窓位置だった。条件3の窓の取出し・処理順とモデル寿命が同時に変わるため、それぞれの寄与の特定は残る。検出からball_points、再投影・速度の棄却までの伝播は比較JSONに保存した。027の8は有効から `REPROJECTION`、71と147〜149は `EXCESSIVE_SPEED` に変わり、037の286は `REPROJECTION` に変わった。027の232は両条件で無効のまま、不採用の理由が `BEHIND_CAMERA` から `INSUFFICIENT_VIEWS` に変わった。

人物の差はDINOの検出box、score、採用行から現れた。人物対応の `player_ids` は000で1要素、027で14要素が変わった。最終の `player_track_ids`、root・heading・SMPLの有効maskと全clipの停止理由は一致。000の261、283、299、306、308、316で各1関節の3D有効maskと棄却理由が変わった。共通有効の座標でも全clipで固定の1 mmを超え、最終sceneの品質判定は3clipとも不合格だった。

## ボールなしの人物3D

人物からcourt_sideを判定していても、ボールありの `camera_alignment` は人物と球を幾何の検証に使う。旧逐次・逐次対照・横断batchの初回から保存済みの校正・実GPUの人物対応・2D poseを読み、ボールなしの設定でカメラ検証・人物三角測量・身体view選択をCPUで再実行した。旧逐次の保存設定は現行schemaに必要な `reuse_clip_embeddings` が未定義のため、最新の逐次対照の `pipeline.yaml` を明示して使った。preflightでこの項目が未定義から `false` になったことを照合した。

9件とも人物のみの幾何検証を通過し、各保存時とカメラ幾何、人物三角測量、身体モデルへの入力が完全一致した。保存済みの身体配置を使って球なしのsceneをCPUで再組立てし、9件の人物配列が各GPU保存結果と完全一致した。旧逐次から新しい逐次の人物3Dは一致し、同一HEADの横断batchで生じた人物3Dの差と6関節のmask差は球を外しても残った。この確認で使ったモデル出力は実GPUの保存済み成果物で、ボール無効設定での全GPU pipelineの再実行は別途未計測。 [プレーヤーのみの再計算](player-only-replay.json)

## sceneと差分動画

今回追加で計測した条件3の9件と逐次対照の3件を公開markerから再読込した。条件3の初回で4カメラともBallの検出が空になったframeは000で69、027で3、037で27あった。動画は各比較の位置差最大、有効maskまたは棄却理由が変わったframeと、条件3の空検出例（000の29、027の227、037の0）を含む6本。各動画でH.264とフレーム数を確認し、027の148では逐次の4観測から条件3の3観測・`EXCESSIVE_SPEED` への変化を確認した。対象全clipが生成に成功し、clip全体の再構成失敗、中断・再開、cache混在の実GPU検証は未実施。 [差分動画のファイル・frame一覧](review-index.json)

job `0-8204898262595246886_2026-11-19T15-22-17-404753113Z_3013697_i1757-serial-fb6255a-first-1` のstdout/stderrは共有queueの `.training_queue/logs/<job-id>.log`。作業treeの `outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/serial-fb6255a-first-1/` に `measurement/terminal.json`・`measurement/worker.log`・`attempts/1-first/recording.json`・`compare-*.json`・`review/*.mp4` がある。
