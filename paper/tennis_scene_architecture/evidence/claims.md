# 主張と根拠の対応表

本文・表の `\ev{ID}{値}` は下表の行を参照します。IDはPDFには印字しません。表の採用行・列と丸めは [表生成スクリプト](../scripts/make_tables.py) にあります。図中の数値は「図の根拠」表から同じデータにたどれます。

固定資料のID、完全なGitコミット、元ファイルのパス、内容のSHA-256は [sources.tsv](sources.tsv) に記録しています。資料は `git show <commit>:<path>` で参照できます。runの入力指紋、設定、実際の実行コミットは、対応する記録の内部にあります。図版の元パスとコピー先、双方のハッシュ、再生成方法は [figures.md](figures.md) に記録しています。

採否は [設計書](../design.md)、[調査の統合コメント](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189580548)、追試の保存値を照合して決めました。[結果を解釈するための規則](../../../knowledge/summary.md#結果を解釈するための規則)

## 本文と表

| ID | 掲載箇所 | 掲載値・指標 | 評価条件・分母・取り出し方 | 固定資料のID |
| --- | --- | --- | --- | --- |
| `sample` | 第1・4章、付録A | 3視点、1,010元フレーム。球821、選手別の関節1,008／993フレーム。SceneResult v2 | Meijiの公開サンプル、`video_000/clip_000`。標準経路の動作確認。動画ごとのSHA-256と公開版、マスク別の数を照合 | `sample-node` `sample-input` `sample-data` |
| `input` | 第3・4章 | 3〜5視点、クリップ設定ではコート観測はフレーム0 | 同期、同FPS・同解像度の固定カメラ。共有コート設定の観測と区別する | `scene-readme` `pipeline-config` `scene-pipeline` |
| `schema` | 第3章、表 `tab:scene-contract` | COCO 17関節、コート14点、画像座標2成分、コート座標3成分。無効座標0、理由0が有効、球は2視点以上。float32・bool・uint8。統合v2・公開v3 | `validate_scene_result_arrays` の形状、型、マスク間制約。時間軸は元フレーム。v3は疎なイベント列とcoverageを追加 | `scene-schema` `scene-readme` `scene-pipeline` |
| `ball-geometry` | 第4章 | 2次運動の整合、頑健な2次LOESS | 標準の `ball_points`。補間と採用実観測のマスクを分離。距離は画像対角長、窓幅はFPSに対応 | `scene-pipeline` `pipeline-config` |
| `geometry` | 第4章 | 幾何支持は2視点以上 | 視点ペアの候補、画素再投影、支持した視点での再推定。人物の足元と外観の対応は別の処理 | `triangulation` `association` `scene-pipeline` |
| `court-corpus` | 第5章「画像と幾何教師の生成」 | 報告版4シーン・8,415枚・141軌道群、生成時の記録8,433枚 | B00〜B03。報告版は分割表の6,594／935／886枚の合計。異なる版を混ぜず、後段実験はそれぞれの分割で記述 | `court-report` `court-audit` `court-render` `court-generation-node` |
| `court-method` | 第5章、図 `fig:court-projection` | 当てはめ32視点、保留16視点 | B00の実画像LINE証拠を地面格子へ投影。図は当時のLINE専用モデルと保存結果。表示視点と集計視点を区別 | `court-report` `court-align` |
| `court-scenes` | 第5章、表 `tab:court-scenes` | 1〜4シーンの実画像KP平均53.9／83.4／43.7／23.6、中央値5.81／67.92／3.68／3.42 px。学習5,500枚、33,160更新、実画像2,211枚、長辺512、seed 42 | 000055。凍結DINOv3-S、B00から順に追加、最終checkpointで旧TCDの共通validation。CSVの `real/kp_*` | `scenes-node` `scene-results` |
| `court-pairs` | 第5章「合成シーン構成」 | KP平均の群内集計は異群33.93、同群83.34 px | 000067。異群4対・同群2対のモデル集計をそれぞれ等重みで平均。路面・シーン特性で分けた各シーン対を、旧TCD実画像2,211枚で採点。CSV `kp_mean_distance_px` | `pairs-node` `pair-results` |
| `court-views` | 第5章、表 `tab:court-viewpoints`、図 `fig:court-viewpoints` | 4シーン各128枚・計512枚、6,000更新、seed 42・43。test全体のKP平均103.50→111.07 px、H成立率32.41→22.19%。全層の画像・動画数と指標は表生成スクリプト | #2633、000083／000084。固定 `real-video-v1`、test 4,372画像・249動画。長辺512換算の可視KP平均。H成立はPROSACが写像を得た画像数。各seed等重み。図は広−狭、対応付き動画bootstrap 2,000回の95%区間 | `court-view-node` `court-view-group` `court-view-results` `court-view-csv` `court-view-prep` `court-view-split` |
| `court-h` | 第5章「共同推定と固定カメラでの運用」 | local01、KPのみから共同最適化で線支持73.9／81.8→99.4／99.8% | 保存KPとLINEの同一推論を使用。順方向は線ごと、逆方向は確率で重み付けし、距離閾値内の支持を集計。独立の校正真値に対する精度は未計測 | `court-h-node` `court-template` `court-report` |
| `court-shared` | 第5章「共同推定と固定カメラでの運用」 | 69クリップ・276カメラ区間で利用可能、159区間に警告 | 000078。固定Hを再推定せず線の変位を別フレームで診断。証拠不足を警告として保持 | `shared-court-node` `shared-court-data` |
| `track-2d` | 第6章、表 `tab:track-2d` | raw IDF1 0.930→0.955。表のraw IDF1・group IDF1・pair F1と照合box数 | 000012、Meiji 3cam開発4本。入力検出を固定。追跡方式により、照合可能なboxとpair採点の対象は異なる | `tracking-node` `tracking-data` |
| `track-merge` | 第6章「カメラ内の追跡方式」 | 40,531検出から26行を除去、IDF1とpair F1は同値 | 000013、同じStrongSORT++＋poseで高IoU重複統合の有無。CSV `merge_off`／`merge_on` 行 | `merge-node` `tracking-data` |
| `selection-3cam` | 第6章、表 `tab:selection-3cam` | 予約の選手P 63.25→100.00%、group IDF1 74.98→99.71%、共通成立2本のpair F1 99.42→99.30%。開発4本・予約3本の各選手P・R、IDF1、pair F1、成立数 | #2634、000015。固定DINO・追跡・姿勢・CLIPで選別規則だけを変更。P・Rは部分参照の既知box、IDF1は重複をまとめた人物・視点・時刻。予約の選手参照ID単位は21,989。向きと校正は各条件で固定 | `selection-node` `selection3` `selection-inputs` `association` |
| `selection-4cam` | 第6章、表 `tab:selection-4cam` | 三角測量到達は画像規則0/69、コート滞在と連結55/69。各条件の有効選手フレームと被覆 | #2634。選手の観測で向きを決める設定のCPU再生。全69件・184,912選手フレームが分母。1点以上の有効関節を持つ選手フレーム／クリップを数える | `selection-node` `selection4` `selection-inputs` |
| `stopped` | 第6章「停止時に失われた2D観測」 | 停止14件、基本42時点。評価可能665枠に検出と追跡あり、選別後581枠。境界の停止9件で次点に実選手 | #2636、000101。4視点の同時点をレビュー。評価可能な選手・視点枠を分母とし、重なりの診断は前後の確認も含む | `stopped-node` `stop-review` `stop-labels` `stop-events` `stop-inputs` |
| `stopped-scenes` | 第6章、図 `fig:stopped-004`・`fig:stopped-020` | clip_004の282フレーム、clip_020の131フレーム、各4視点 | 元フレーム番号。元DINO行、追跡ID、選別結果を保存画像へ重畳。clip_004はcam3の前後も記載 | `stop-figures` `stop-events` `stop-inputs` |
| `stopped-replay` | 第6章「停止時に失われた2D観測」 | 14/14停止再現、DINO行除去0/8回復、group統合0/2、次点除外7/9人物対応通過、うち5件で各人2視点以上 | #2636。保存特徴によるCPU再生。回復・通過は介入ごとの数で、5件はレビュー時点でIDを確認 | `stopped-node` `stop-replay` |
| `track-3d-3cam` | 第6章、表 `tab:tracking-3d` | 予約未見IDF1 92.07→80.42%、選別再現率85.30→67.25% | #1739/#2446、000106。予約未見3本の人物IDを採点。新経路の時間IDは元画像へ射影。共通の人手参照とIoU条件。旧で校正停止の1本は新で校正補完 | `track3d-node` `track3d3` |
| `track-3d-4cam` | 第6章、表 `tab:tracking-3d` | 完了52/69→69/69、人物停止14→0、全員の有効配置64.63→90.74%。共通55本の再投影残差7.89→24.86 px | #1739/#2446、000106。4cam全69本。完了は全選手それぞれに身体と配置があり、球3Dがあるクリップ。残差は未採用の観測視点で、clip中央値の中央値、方式ごと78,664／82,972点。計測版は開発枝、現行の明示選択入口は別資料 | `track3d-node` `track3d4` `route` |
| `ball-config` | 第7章「標準モデル」 | i935はMDD 2チャネル、窓8フレーム | 既定 `conv_next_unet` の設定。差分特徴と重複窓の扱いは出力契約に従う | `pipeline-config` `scene-pipeline` `ball-mdd` |
| `ball-x3d` | 第7章「標準モデル」 | X3D／CSN＋DPTの明示設定は32フレーム。固定幾何で有効球3Dは113／356と123／356元フレーム | 000043。同じMeiji 4cam映像の有効検出数は視点・方式で異なる。元画像と入力解像度を区別。固定幾何再生はclip_000のcam1〜3でX3Dの校正を両方式へ渡したCPU診断。通常実行とは別の記録 | `x3d-node` `x3d-data` `scene-pipeline` |
| `ball-origin` | 第7章「外部注釈」 | 初版69 JSON、2026年11月24日01:50:43 UTC取得。同日20:27 JST開始のGPT-6.2 Sol/maxで3本（012、015、026）を再注釈 | #2637、000097。初版ZIPとメンバーのバイト同一性。初版の生成モデル・実行IDは未確認。rev1のモデル・時刻・コード終了値・各出力ハッシュはcampaign台帳と照合 | `footprint-node` `ball-origin-data` |
| `ball-footprint` | 第7章、表 `tab:ball-cameras`、図 `fig:ball-footprints` | 69クリップ、1920×1080・約59.94 fps。元画像2,121/2,208で径4.00 px、512×288では360/2,208で径1.76 px。全サイズで計測339件。視点別の成立数、径、伸びも掲載 | #2637、カメラごと各clipの画面内observedから固定抽出。3画像サイズ（中間1280×720）を実際に縮小。残差成分を測定画像pxで集計。量ごとに測定成立例と共通集合の母数を区別 | `footprint-node` `ball-media` `footprints-data` `visibility-summary` |
| `ball-visibility` | 第7章、表 `tab:ball-cameras`、図 `fig:ball-visibility` | 4,136/184,912（2.24%）座標範囲外、推定4,125。目視64件の可視27、遮蔽2、識別不能11、画面外1、範囲外座標16件のうち退出未確認15。各視点46,228座標、16監査 | #2637。単位はカメラ・元フレーム。座標範囲の全数診断と見え方の層化標本は別集合。視点・区分ごとの判定は保存表 | `footprint-node` `visibility-summary` `visibility-data` `visibility-review` `visibility-totals` |
| `ball-capacity` | 第7章、表 `tab:ball-history` | 最小約8.0M、平均35.38・P95 169.69元画像px。表の全容量条件の規模・誤差・更新 | 000088。共通長系列validationの85,473 observed、T64、seed 42、query学習12,000更新。CSVの `series=capacity` | `ball-capacity-node` `ball-history` |
| `ball-length` | 第7章、表 `tab:ball-history` | DPTとqueryの32→64フレーム、観測85,473件、平均・P95、累積更新数 | 000068。固定の長系列validationで同じ採点単位。学習履歴と累積更新数も異なる。CSVの `series=long_sequence` | `ball-length-node` `ball-history` |
| `ball-camera-stress` | 第7章「容量・系列長と人工カメラ変動」 | Fullの平均15.4→32.2元画像px | 000060。clean 155,080／人工カメラ変動154,977フレーム・FPSの組。変換後画面外を除外し、FPS別の平均を等重みで集計 | `ball-stress-node` `ball-stress` |
| `ball-intervention` | 第7章、表 `tab:ball-intervention` | アンカー144、observed 88、元画像20 px以内でX3D 76→2、i935 73→0、形状7件。シャッフルは55・67件。表は平均・P95とFPR・FNRも掲載 | #2637、000098。X3D T32と形状は1,280×720、i935 T8は512×288。通常・反復・順序入替で学習済み重みと閾値を共通に使用。FPRの分母は確認不在56件、FNRの分母は観測88件 | `intervention-node` `ball-anchors` `ball-intervention` |
| `ball-train-plan` | 第7章、表 `tab:ball-training`、図 `fig:ball-training` | 512×288、seed 42、学習30,478パック、4,700更新×128枚＝601,600入力フレーム。T=1 RGB、T=8・32 RGB＋MDD | #2697、000099。X3D／CSN＋DPTを共通とし、各条件をスクラッチで学習。初期重みの共通部分と更新ごとの入力順・教師数を照合 | `ball-train-node` `ball-train-plan` `ball-train-check` |
| `ball-train-val` | 第7章、表 `tab:ball-training`、図 `fig:ball-training` | validation 117,358件（observed 85,473）。20 px以内率1.66／40.90／44.23%、平均226.75／134.06／125.74 px、中央値177.02／42.48／31.68 px、存在F1 74.07／82.02／82.63%。時系列同士の差3.33ポイント | #2697。位置採点は閾値前の全observed。存在は同じvalidationで（FPR＋FNR）/2を最小にする閾値。独立testは未計測 | `ball-train-node` `ball-train-csv` `ball-train-data` |
| `ball-train-sources` | 第7章、表 `tab:ball-training`、図 `fig:ball-training` | 20 px以内率はChat 1.29／45.54／49.20%、Meiji 3cam 2.66／27.36／29.77%、TrackNet game9 1.37／65.18／69.10% | #2697、同一validationを収録元別に分け、observed 61,369／23,007／1,097件で集計。3入力条件の順 | `ball-train-node` `ball-train-csv` `ball-train-data` |
| `ball-train-anchor` | 第7章、表 `tab:ball-training`、図 `fig:ball-training` | 144アンカー、20 px以内2/88（2.27%）→40/88（45.45%）→45/88（51.14%）。時系列同士の差5.68ポイント | #2697、#2637と同じ固定アンカー。observedが位置の分母、存在は固定validationの閾値を転用 | `ball-anchors` `ball-train-node` `ball-train-csv` `ball-train-data` |
| `execution-plan` | 第8章「依存の宣言」 | 69クリップ、CPU事前計画2,622ノード | 000072。人物のみ、動画共有コート、選手の証拠で向きを決める設定。GPUモデル読込とノード実行を伴わない計画検査 | `plan-node` `plan-data` |
| `execution-shared` | 第8章「モデルの寿命」 | 3クリップで管理対象7資源が各1ロード・1解放 | 000064。1計画・1runtimeの球あり条件。CPU AFLinkは別管理。#2696の管理対象6資源と区別 | `shared-node` `shared-data` |
| `execution-fill` | 第8章「モデルの寿命」 | ViTPose 99.98%、CLIP-ReID 99.96%、HMR2 99.86% | #2363、000083、全69クリップ。横断dispatchの規定の上限に対する実充填。DINOとGVHMRの処理単位とは区別 | `full-node` `full-performance` |
| `execution-kernel` | 第8章「TensorRT」 | ViTPose-Hの遅延中央値54.423／27.197／8.468 ms、FP32基準の関節座標差4,012点のP95 0.02471、最大0.30197 px | 000077。同じclipから固定したcropの同期済みモデル単体評価。順にPyTorch FP32、PyTorch FP16、TensorRT FP16。座標差はTensorRT FP16のデコード済み元画像座標 | `kernel-node` `kernel-data` |
| `lovo` | 第9章、表 `tab:lovo` | Meiji 3camの1本、評価区間[505,1010)。基準方式825/1,462、共通768/1,462点で中央値8.97・P90 25.40 px。同じ集計をLOESSなしにも掲載 | 000033。カメラを1台ずつ球の復元から外し、留保視点に射影。全視点由来の固定カメラを使う。スコア可能なobservedとestimatedを含み、共通はoracleを含む全比較方式の積集合。元画像px | `lovo-node` `lovo-data` `lovo-inputs` `scene-eval` |
| `pseudo-reference` | 第9章「幾何整合と被覆」 | 6,844画像、参照人物26,276件。有限434,946関節点の中央値3.42 px | 000084。保存SMPLを同じカメラ幾何で射影した疑似参照と保存ViTPoseの接続検証。全画面検出の適合率や独立3D参照ではない | `pseudo-node` `pseudo-data` `pseudo-inputs` |
| `full-coverage` | 第9章、表 `tab:full-video` | 生成55件で有効関節点2,270,935/2,364,496（96.04%）、SMPL 133,235/139,088（95.79%） | #2363、000083。生成済みクリップ・全選手・元フレーム。関節は部位別のマスク、SMPLは選手フレーム単位のマスク | `full-node` `full-clips` |
| `full-timing` | 第9章、表 `tab:full-video` | 55/69生成、停止14、CLI 43,858.46秒（12時間10分58秒）、VRAM 1.76 GiB、RSS 6.16 GiB | #2363。RTX 5060 Ti、固定コート、球無効、選手証拠、横断TensorRT FP16。起動から集計と書き出しまで。NVMLデバイスとprocessのサンプル最大、bytesをGiBへ換算 | `full-node` `full-data` `full-performance` `full-condition` |
| `full-stops` | 第9章「登録動画の一括処理」 | 人物対応13件（次点境界9、片側不足2、割当未決2）、コート向き1件 | #2363。全試行の停止段と理由コード。#2636の観測介入は別の評価 | `full-data` `full-clips` |
| `runtime-setup` | 第9章「実行系の統制アブレーション」 | 3クリップ（000・037・027）、4カメラ、各カメラ計883元フレーム、C1〜C4各1回 | #2696、000107。実行版 `2dbd8c355faa2d530b8b0e17164f691e60a57d07`、RTX 5060 Ti、固定コート、独立した空store、球無効。資産準備は計時の外 | `runtime-node` `runtime-condition` |
| `runtime-metrics` | 第9章、表 `tab:runtime`、図 `fig:runtime` | CLI 2,186.31／1,686.15／1,706.81／1,319.31 s、C1比1.00／1.30／1.28／1.66倍。VRAM 4.49／4.49／4.49／1.76 GiB、RSS 7.61／7.76／8.16／3.77 GiB。読込・解放18/18→6/6。直前差は順に短縮500.17、増加20.66、短縮387.50秒。C3の主要cropバッチ充填率99.77%以上 | #2696、C1逐次PT16から共有、横断バッチ、TRT16へ移行。比と増減は丸め前の値で算出。C4の約61% VRAM減・約50% RSS減はC1基準（C3基準では60.76%・53.73%）。NVML/RSSは0.5秒採取。各条件で全3クリップ生成 | `runtime-node` `runtime-csv` `runtime-data` `runtime-condition` |
| `runtime-parity` | 第9章、表 `tab:runtime-parity`、図 `fig:runtime` | 12選手ID。C2完全一致、C3最大0.079 mm。C4共通59,533点、中央値1.81 mm、P95 9.06 mm、最大0.698 m。SMPL有効3,496→3,495 | #2696。C1と対応する共通の有効関節だけでL2距離を測定。全条件の関節フレーム有効数3,508。表では元のmをmmへ換算し、最大値は小数第3位まで表示 | `runtime-node` `parity-csv` `parity-data` |

## 図の根拠

図番号は原稿のラベルで指定します。[元ファイル・コピー先とハッシュ](figures.md)

| LaTeXラベル | 本文との対応と図中の量 | 元データのID |
| --- | --- | --- |
| `fig:pipeline` | 第4章。標準経路の依存を示す模式図。入力・出力の型と球観測の採用 | `scene-pipeline` `pipeline-config` `scene-schema` |
| `fig:court-ground` | 第5章。高さ帯の候補とRANSAC支持、確定スケールでの点群座標（m） | `court-align` `court-report` |
| `fig:court-projection` | 第5章。実画像のLINE確率と、当てはめ視点の地面格子上の集約量。条件は `court-method` | `court-align` `court-report` |
| `fig:court-alignment` | 第5章。B00〜B03各1視点のレンダリングと規格コートの再投影。画像IDとカメラ情報 | `court-render` `court-report` |
| `fig:court-viewpoints` | 第5章。KP平均の差（長辺512換算px）、H成立率の差（ポイント）、seed別・平均・動画bootstrap区間。条件は `court-views` | `court-view-results` `court-view-csv` |
| `fig:court-real` | 第5章。local01〜04の指定実画像、TCD argmaxと共同最適化による保存推論 | `court-inputs` `court-detection` `court-template` |
| `fig:court-drift` | 第5章。各シーンの距離軸（m）、地面の高さ差（後半−前半、cm）、共通セル数。両区間に各5点以上ある0.5 mセル、色域±20 cm | `court-drift-data` `court-report` |
| `fig:stopped-004`・`fig:stopped-020` | 第6章。元フレーム、視点ID、t（トラック）、g（候補）、r（検出行）。条件は `stopped-scenes` | `stop-figures` `stop-labels` `stop-events` |
| `fig:ball-footprints` | 第7章。測定画像pxでの径・伸びの累積分布、視点別の計測成立率。条件は `ball-footprint` | `footprints-data` `visibility-summary` `ball-media` |
| `fig:ball-visibility` | 第7章。座標中心の範囲外率、外部可視性フラグ、層別目視の判定数。条件は `ball-visibility` | `visibility-data` `visibility-summary` `visibility-totals` `visibility-review` |
| `fig:ball-training` | 第7章。全validationの元画像pxでの位置精度、存在F1、FPR、FNR、収録元別の20 px以内率。アンカーは本文の表で別に集計。条件は `ball-train-plan`、`ball-train-val`、`ball-train-sources`、`ball-train-anchor` | `ball-train-csv` `ball-train-data` |
| `fig:execution` | 第8章。クリップを単位に計画・実行・公開する模式図。計画、runtime、storeの責務 | `scene-pipeline` |
| `fig:runtime` | 第9章。条件別CLI時間、段階・モデル別の時間、ピークメモリ、モデル読込回数。条件は `runtime-metrics` | `runtime-csv` `runtime-data` `parity-csv` `parity-data` |

## 調査・委任時の記述との差分

- #2633は単一シーンからの抽出という指定箇所があった。比較の準備記録と合意コメントに従い、B00〜B03の同じ4シーンを使った小規模学習と記載した。[第5章の合意](https://github.com/Motoki0705/tennis-lab/issues/2628#issuecomment-7205422706)
- #1739の7.89→24.86 pxは、Meiji 4camで両方式の残差が求まった55本の比較である。Meiji 3cam予約未見3本では時間IDを採点した。実行時の開発枝はその後に明示選択式の経路として統合されている。本文は開発時点と統合後を明示した。[採点の元記録](../../../knowledge/runs/run-meiji-multiview-scene/README.md) · [経路の統合PR](https://github.com/Motoki0705/tennis-lab/pull/2470)
- #2696のRSS低下約50%は、逐次C1の値を分母とした。C3からC4への低下率は53.73%。固定条件の時間比較と、異なる全件実行の所要時間はそれぞれの計測単位で記載した。[実行系の比較記録](https://github.com/Motoki0705/tennis-lab/blob/f3ba146f57c07d8f1e168cb03fc6e71b9b9fdac1/knowledge/runs/run-i2696-exec-system-ablation/README.md)
