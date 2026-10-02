---
id: run-i935-source-tail-audit-r27-20261001
type: run
task: ball_refiner
sequence: 26
recorded_at: '2026-10-01'
title: source/cacheの裾差を候補・入力・block bootstrapで切り分ける
provider: codex
status: done
issue: 935
date: '2026-10-01'
config: {clip: meiji/video_000/clip_010, cameras: [cam0, cam1, cam2], device: cpu}
metrics:
  camera_frames: 810
  winner_component_changes: 146
  cam2_cpu_raw_p90_px: 158.22421283420547
  cam2_cpu_resize_only_p90_px: 78.58804117807715
  cam2_cpu_reencoded_p90_px: 55.16200120068717
  pooled_block33_ci95_px: [-34.3246743248061, 93.37947172645046]
artifacts: {run_dir: knowledge/runs/run-i935-source-tail-audit-r27-20261001}
parents: [run-i935-source-b-gate-r26-20261001]
relations: []
papers: []
tags: []
---

## 結論

**(b)を支持する。より正確にはJPEGだけでなく、中間720p縮小も含む入力経路差が、候補集合・順位・成分選択を変えて裾へ増幅される。**
同じpipelineコードで入力画素だけをstoreに合わせるとcam2のp90と全270frameの最大weight成分選択がcacheへ戻った。
一方、pooled +46 pxがsampling noiseの外にあるとはblock bootstrapから示せず、(c)は差の大きさ・一般化の不確実性として残る。
(a)のframeずれ・PTS・窓・色順・正規化・dtype・model取り違えを支持する証拠はなく、production bug修正は行っていない。
任意の未監査bugの不存在を証明した、またはsampling noiseだけが原因だったとは主張しない。
default・固定Bゲート・共分散倍率1.8125148752087792を維持する。Bは元の654frameに対してFAILのまま。

## 全frame比較

3camera×270frameのframe/PTS/実秒、検出器8frame/stride4とrefiner33frame/stride16の採用窓は一致した。
検出器の同順位native cell一致は859/6480候補slotのみ。20 source px以内の空間対応では
3306/6480候補が対応し、758/810frameで対応候補の順位が入れ替わる。
この対応は同一物体の識別ではなく、候補変化を調べるための最大対応数・最短距離の診断である。

| camera | 最大weight成分の変更 /270 | 両経路でanchor成分が最大 | そのanchorの空間対応変更 | 自由成分が最大 cache→source |
|---|---:|---:|---:|---:|
| cam0 | 60 | 202 | 23 | 59→22 |
| cam1 | 24 | 236 | 19 | 27→22 |
| cam2 | 62 | 191 | 10 | 42→60 |

全行の正本は[frames.csv](../../runs/run-i935-source-tail-audit-r27-20261001/frames/frames.csv)、
候補の位置・score・順序は[candidates.csv](../../runs/run-i935-source-tail-audit-r27-20261001/frames/candidates.csv)、
全成分の平均・weight・anchor slotは[components.csv](../../runs/run-i935-source-tail-audit-r27-20261001/frames/components.csv)。
NPZには局所patchと全GMMの基礎fieldも残した。対応判定・無効候補・modelと同点順序の一致を3テストで検証、ruff/mypy成功。
入力hashを実行前後に照合。GPU未使用、学習を行わないためTensorBoard曲線はない。

## 入力・前処理の照合

全810frameでOpenCVのBGR decodeとPyAV bgr24が全画素一致し、PyAVの実PTSはstoreと一致。
mp4 decode→INTER_AREA 1280×720→JPEG quality90で、保存store JPEGを**810/810 byte一致**で再現した。
検出器入力も810/810でbit一致。読込の色順・frame順・別decoderの違いを原因とする証拠はない。
保存画像と元動画の最終入力は同じBgrToTensorTransform（INTER_LINEAR 512×288、RGB、float32 /255）を使う。
同じcheckpointのadapterが同じImageNet正規化を一度適用した後、同じMDD差分特徴へ変換する。

| camera | 最終RGB差 MAE / RMSE (0..255) | resizeのみ RMSE | JPEGのみ RMSE |
|---|---:|---:|---:|
| cam0 | 2.333 / 4.424 | 4.085 | 1.560 |
| cam1 | 1.385 / 2.181 | 1.700 | 1.298 |
| cam2 | 1.518 / 2.730 | 2.321 | 1.380 |

JPEGだけでなく、1080pから720pへの中間縮小が実質的な入力差である。
列のRMSEは加算分解ではなく、それぞれ二つの入力を直接比較した値。
全行とpixel hashは[pixels](../../runs/run-i935-source-tail-audit-r27-20261001/pixels/summary.json)。
CPU 152.0秒、最低host空き13.79 GB。新規画像・動画は保存せず、結果CSV/JSONだけを保存した。
媒体差の因果性を次のdetector再推論で検証した。
cam2の再encode介入は完了した。**既存のBallDetectionModule._predict_videoを呼び、readerが返す画素だけを変更**。
全67窓についてstore側のRGB batchとmodel入力の一致を確認し、同じe9/anchored seed42/固定倍率をCPUで推論した。
cacheに対しnative cellは2150/2160候補slot、全順位は265/270frameで一致。
CPU/CUDAの小さな数値差により低順位候補の入替が残り、同順位座標の最大差だけで一致性を判断できない。
cam2 observed p90はpipeline座標のままで55.162 px、store座標規約も合わせると55.189 px。
保存cacheの55.047 pxと近く、元動画158.177 pxから裾が戻った。自由成分最大も60→42でcacheの42と一致。
座標規約の相違はstore幅比で戻すかsource端点で戻すかで、uv倍率[0.999739,0.999537]、最大0.5 px/軸。
これを変更せず入力だけを変えた場合でも裾が戻るため、今回の大きな差を説明する主因ではない。

正本は[CPU isolation](../../runs/run-i935-source-tail-audit-r27-20261001/isolation/cpu-v3/cam2-reencoded/summary.json)。
CPU約210秒、GPU未初期化。初期の診断runnerでruntime設定からYAML設定へ変換する際のdevice/enable key errorが2回あり、
いずれもdetector推論前に停止。scriptを修正して再実行し、失敗logも残した。productionのbugではない。
raw/resizeのみの同一CPU対照も完了。以下は全270frameを推論したうち同じ217 observed GTの結果で、scale再fit・score閾値・frame選別はない。

| cam2経路 | median px | p90 px | 位置NLL nat | 自由成分最大 /270 |
|---|---:|---:|---:|---:|
| 保存cache CUDA | 3.157 | 55.047 | 5.664 | 42 |
| 保存source CUDA | 3.270 | 158.177 | 6.158 | 60 |
| sourceを同じCPUで実行 | 3.274 | 158.224 | 6.158 | 60 |
| 中間720p縮小のみ（JPEGなし）CPU | 2.396 | 78.588 | 5.336 | 36 |
| 中間縮小＋JPEG90 CPU | 3.357 | 55.162 | 5.671 | 42 |

raw CPUは保存source CUDAと最大成分が270/270一致、再encode CPUは保存cache CUDAと270/270一致。
中間縮小だけでp90は大きく戻り、JPEGを加えるとcacheに近づく。非線形なので各段の寄与の加算分解ではない。
同じRGB前処理・正規化を通っても、MDD入力のraw対cache最大差は0.99350、再encode対cacheは0。
小さいRGB差が局所的な時間差分特徴で大きくなり得ることも確認できる。
anchor成分の同index平均が100 px超ずれる1,094組は全て、そのindexのanchorの空間対応が変わっていた。
同じ候補へ対応し直すと平均距離のp90はcamera別12.69–14.54 pxであり、strict means最大差を単なる座標変換誤差と解釈できない。
ただし順位の交換と候補自体の出現/消失、score/patch変化を別々の因果効果として分離した実験ではない。

数値の正本は[conclusion/summary.json](../../runs/run-i935-source-tail-audit-r27-20261001/conclusion/summary.json)、
全observed行と[比較図](../../runs/run-i935-source-tail-audit-r27-20261001/conclusion/tail-audit.png)も保存した。
CPU3条件の合計679.997秒、peak RSS最大3.325 GB、最低host空き13.419 GB、CUDA初期化なし。

## p90のsampling noiseと末尾

observedだけを詰め直さず、各cameraの元270frameを連続blockに切り、同じ抽出indexを両経路へ適用してからobservedを選ぶ。
10,000反復、seed1729、linear quantile、percentile 95%区間。長さ8/16/33/66frameを比較した。
最初のnon-wrapping moving blockは端のframeの採用率が低くなるため、末尾が重要な今回は感度解析として保持する。
主に解釈するdisjoint block法は短い端blockも丸ごと復元抽出し、全frameの期待出現回数を1に保つ。
開始位置0と半blockずらし、camera独立抽出と同時刻cameraを一緒に抽出する感度解析も全て残した。

| block長 | pooled Δp90 95%区間 px（開始0、camera内独立） |
|---|---:|
| 8 | [-15.87, 90.16] |
| 16 | [-25.05, 93.06] |
| 33 | [-34.32, 93.38] |
| 66 | [-35.02, 94.49] |

16条件全てで0と観測された+46.34 pxを含む。**pooled +46 pxがsampling noiseの外にあるとは示せない。**
33frame・開始0の同期camera抽出も[-42.58,71.38]。同条件camera別ではcam0 [-76.99,5.10]、
cam1 [-61.58,119.09]、cam2 [3.20,149.58]で、cam2の局所的退行とpooledの不確実性は両立する。
これは4.5秒の単一clip内の探索的診断で、独立clipに一般化できる区間でもp値でもない。
block母数は少なく末尾も非定常で、p90の標本分布は偏っている。CIの広さを使って固定Bゲートを変更しない。

cam2の217–265（49/49 observed）では自由成分最大が14→39frame、元sourceのpooled p90=140.69 pxを超える行が5→24。
この49行だけのsource誤差をcacheへ置く**事後的な数値診断**ではpooled p90=72.25 px、差=-22.10 px。
両経路から除く場合も差=-20.89 pxとなる。frame230は最大成分0→3、誤差1.90→89.75 px。
frame250は3→1、131.34→579.20 px、frame265は0→3、3.26→173.90 px。
これはgate再採点や、frameを除外する提案ではなく、裾の寄与の説明である。

正本: [bootstrap summary](../../runs/run-i935-source-tail-audit-r27-20261001/noise-balanced-v2/summary.json)、
[末尾49行](../../runs/run-i935-source-tail-audit-r27-20261001/noise-balanced-v2/cam2-217-265.csv)。
一部の長blockでcamera単独のobservedが0件の反復は未定義件数として明示し、0誤差へ置換しない。pooledの未定義は0件。

## 次の選択肢とコスト（未選択）

run指示に従い、以下のどれを採るかは選ばない。現default維持は今回の固定制約であり、対策Cの長期採用判断ではない。

| 選択肢 | 変更・検証の範囲 | コストの目安・未確認事項 |
|---|---|---|
| 修正して同じgateで再確認 | 今回、直すべきindex/色/窓bugは見つからなかった。入力経路を修正対象にするなら、store同等前処理を明示optionとして実装し、入力契約・identity・テストを更新。既存e9/refiner/倍率は固定して3camera Bを再実行 | 実装・テストは小規模だが全camera受入は未検証。r25実測130秒/peak device3.0 GB/report32 MBを基準にGPU 3–8分、上限20分、8 GB未満、出力0.1–0.5 GBを見積もる。まだ投入しない |
| mp4 decodeに揃えた学習証拠を再構築・再学習 | 元MP4を持つMeiji/chatのreaderを直接decodeに統一。TrackNetの元JPEGは別媒体として明示。e9・split・anchored設計を保持しcacheを版分け、seed/較正/同一gateを再検証 | r17の全329clip/145,767frameは49.4分・NPZ116 MB。再生成50–70分＋12k学習15–25分/seed（seed44実測18.5分、出力0.67 GB）。3seed等で計約1.5–2.5 GPU時間、出力2–3 GB程度＋compiler余裕。入力から実測前の見積。新checkpointは旧較正artifactを流用できず別途確認が必要。今回20分grantでは実行しない |
| 現defaultを当面維持 | ft-e13＋旧refinerを保持し、この差を未解決として残す | 追加GPU/再生成diskは0。新設計の改善をdefaultへ反映する時期は遅れる。文脈追加やgate緩和で今回の不一致を覆い隠さない |

時間・容量の根拠は[e9 cache](000015-run-i935-evidence-mixed-e9-trainval-r17-20260930.md)、
[seed44](000021-run-i935-seed44-retry-r24-20260930.md)、[元動画check](000024-run-i935-source-check-retry-r25-20260930.md)。
新しい入力版の性能・全3cameraでの再encode gate通過・独立clipへの一般化は未検証。

## 検証・再実行・範囲

13 tests（新規7＋固定Bゲート6、pytest -n4）、5診断script＋testのruff/mypy成功。
各段階の入力hashを前後で照合し、診断runnerのsetup2失敗とbootstrapのobserved0件停止もlogに残した。
再実行は[command.txt](../../runs/run-i935-source-tail-audit-r27-20261001/command.txt)を参照し、各scriptに新規絶対`--output`を渡す。
新規bundleは約32 MBで5 GB予算内。元data/outputs削除0、GPU job投入0、独立validator指定・試行・完了0。
video_001のframe/推論は未使用、#964/#936のbranch/worktree/jobは未変更。#964未完として文脈追加/rebaseを行わない。
このrunの診断は完了したが、#935の文脈ablation等の未完項目を完了扱いしない。
