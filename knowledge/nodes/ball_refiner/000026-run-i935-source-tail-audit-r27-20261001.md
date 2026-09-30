---
id: run-i935-source-tail-audit-r27-20261001
type: run
task: ball_refiner
sequence: 26
recorded_at: '2026-10-01'
title: source/cacheの裾差を候補・入力・block bootstrapで切り分ける
provider: codex
status: running
issue: 935
date: '2026-10-01'
config: {clip: meiji/video_000/clip_010, cameras: [cam0, cam1, cam2], device: cpu}
metrics: {camera_frames: 810, winner_component_changes: 146}
artifacts: {run_dir: knowledge/runs/run-i935-source-tail-audit-r27-20261001}
parents: [run-i935-source-b-gate-r26-20261001]
relations: []
papers: []
tags: []
---

## 考察 / Findings

run27の最初の節目。固定Bゲートの失敗理由を調査し、default・gate・共分散倍率は変更しない。
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

## 入力・前処理の照合（節目2）

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
これらは媒体差の証拠であり、入力差が実際のp90を生んだかは次のdetector再推論で調べる。
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
raw/resizeのみの同一CPU対照とbootstrapは進行中。現段階では(b)を強く支持するが、最終結論は対照と併せて記録する。

## p90のsampling noiseと末尾（節目4）

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
