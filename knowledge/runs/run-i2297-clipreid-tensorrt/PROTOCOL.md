# 単体推論の比較プロトコル

対象は #2278 の Phase 1。条件0を同じチェックポイント・入力・デコード設定で作り直し、条件1、2、3a、3bと比較する。計測表には run、実装・重み・入力の識別子、GPU/driver、ツール版、実行条件を紐づける。ONNX変換、unsupported op、範囲外shape、エンジン実行の失敗は明示して停止する。ONNX Runtimeの役割はPTQの統計収集に限る。

| 条件 | 計算 |
|---|---|
| 0 | eager PyTorch FP32。CLIPはmatmulとcuDNN convolutionのTF32を無効にする |
| 1 | eager PyTorch FP16。CLIPは重みと入力をFP16に明示cast。ViTPose/DINOで既存autocastを使う場合は、残る演算dtypeを記す |
| 2 | TensorRTのstrongly typed FP16グラフ。FP32のreductionなどが必要ならその範囲を明示 |
| 3a・3b | FP32のONNXを固定キャリブレーション入力でModelOpt PTQ。INT8・FP8 E4M3のQ/DQを挿入し、TensorRTでビルド。Q/DQ外はFP16を基本とし、IO・計算 dtype と実行layerを記録 |

## TensorRTの実行精度

主要な `MatMul`、`Gemm`、`Conv` をモデルごとに列挙し、条件別の期待精度を実行前に固定する。CLIP-ReIDは24個のattention行列積と48個のTransformerの線形演算に加え、入力のpatch embeddingと末尾projectionを検査する。下表は #2297 の初回計測の条件を示す。

| 条件 | CLIP-ReIDの主要演算 | 高い精度を許容する範囲 |
|---|---|---|
| 2 | Transformerの線形・attention 72演算、patch `Conv`、末尾 `MatMul` の計74演算をFP16 | LayerNorm・softmax・reductionやcastにFP32を許容 |
| 3a | Transformerの72演算とpatch `Conv` をINT8 | 末尾 `/visual/MatMul` はQ/DQの対象外でFP16。LayerNorm・softmax・reduction・cast・入出力にFP16/FP32を許容 |
| 3b | Transformerの72演算をFP8 E4M3 | patch `/visual/conv1/Conv` と末尾 `/visual/MatMul` はQ/DQの対象外でFP16。LayerNorm・softmax・reduction・cast・入出力にFP16/FP32を許容 |

ビルド時に `DETAILED` を指定し、実行層のinspector JSONで各ONNX演算を `Metadata` から対応付ける。GEMM・Convは実行タクティクス名に含まれる両オペランド精度と、inspectorに現れる活性入力dtypeを照合する。FP32/INT32の積和累積や、出力・融合後段のcastに使う精度はオペランドの判定から除く。CLIPの `_gemm_mha_v2_` は2つのattention MatMulを束ねる実行層で、3つのQ/K/V入力dtypeを照合する。inspectorで分かるのはこの融合層の入力形式までで、内部の命令ごとの精度は記録対象外とする。確認できた実行層数とONNX演算数を精度別に記録する。主要演算にFP32や未承認の別精度が現れた場合、対応層や精度の根拠が欠ける場合は非ゼロで停止する。タクティクス表記が変わる場合やdynamic shapeを追加する場合は、対象shapeのinspectorで検証方法を確かめる。 [NVIDIAのEngine Inspector仕様](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)

### ViT 3モデルのPTQ設定

CLIP-ReID・ViTPose-H・HMR2には、FP32 ONNXを校正用 `clip_027` で ModelOpt max 校正し、下表の重み付き演算へQ/DQを挿入する。ViTPose-Hの校正時は各cropと水平反転を同一バッチに並べる。

| 条件 | 量子化対象 | ONNXの対象演算数（CLIP-ReID / ViTPose-H / HMR2） |
|---|---|---:|
| 3a `attention-out-max` | 画像backboneの各Transformerブロックでattentionの出力投影1個をINT8 | 12 / 32 / 32 |
| 3b `linear-max` | 画像backboneの各TransformerブロックでQKV投影・attention出力投影・MLPの前後2個の線形演算をFP8 E4M3 | 48 / 128 / 128 |

対象外の線形演算・畳み込み・入力が両方可変のattention MatMulはFP16とする。残差Add・LayerNorm・GELU・softmaxは直接のQ/DQ対象から外し、reductionやcastにはFP32を許す。HMR2のSMPL headはFP16で、`_gemv_mha_v1_` 融合の3入力と、長さ1のself-attentionのMatMulを積に置換するタクティクスの2入力およびshapeを照合する。実行層で見つからない演算は、ONNXグラフで画像入力に依存しない場合に限って別途記録する。設定の根拠は [CLIP-ReIDの校正・層別比較](../run-i2303-vit-tensorrt/int8-diagnosis.json) にある。

TensorRTのcuBLAS GEMVタクティクスでは、累積精度の前に並ぶ2個のオペランド精度を読む。ViTPose-Hのattention出力投影が残差Addと融合した場合は、同形状のFP16残差とINT8活性化の2入力、投影および2個のDequantizeLinearのメタデータ、GEMMの両オペランドのINT8を照合する。

## 入力とshape

評価は #2238 と共通の `clip_000` と `clip_037`。固定8画像のframe 178、146も4台分を含める。PTQは別区間の `clip_027`。同じコート・4方向のカメラから実推論の分布を取れ、既存のplayer evidenceから観測boxを固定できるため選んだ。clip ID、4台のMP4のSHA-256、同一収録元の各カメラの包括frame区間で非重複を確認する。同じ会場におけるキャリブレーションであり、選手や会場単位の汎化は測らない。

| split | clip | カメラ | 0始まりのclip内frame |
|---|---|---|---|
| 評価 | 000 | cam0–cam3 | 32, 73, 112, 178, 221, 260, 301, 328（全区間 0–355） |
| 評価 | 037 | cam0–cam3 | 22, 52, 101, 146, 172, 207, 243, 270（全区間 0–291） |
| PTQ | 027 | cam0–cam3 | 8, 23, 37, 52, 66, 81, 95, 110, 124, 139, 153, 168, 182, 197, 211, 226（全区間 0–234） |

座標・観測行・収録元区間・媒体digestは [固定入力一覧](inputs.json)、再展開したテンソルと非重複の検査値は [tensor lock](tensor-lock.json) を参照。

CLIPは `OpenCVVideoFrameReader` でBGRを始端からdecodeする。選択済みtrackの実観測かつappearance有効なdetector boxを `appearance.sampling.crop` で四捨五入・clipし、RGBに変換して256×128へresizeする。入力のFP32 RGB `[0,1]` に `(x-.5)/.5` を適用した配列を凍結する。全条件で `(1,3,256,128)`。転送・キャストは計測直前に済ませ、時刻計測はvisual towerとCLS/projectionのconcatまで含む。L2正規化は全条件で共通に精度比較時に行う。評価cropのIDは #2238 FP32のplayer associationからclip内ラベルを取得する。キャリブレーションは #1752 のclip_027 replayの選手候補boxを使用する。

後続は同じclip、camera、frameと必要な上流artifactを固定して、現行前処理からテンソルを採取する。以下のフック位置で、キー名・shape・dtype・順序・mask・前処理設定・元artifactの識別子を記録する。キャリブレーション側でも同じ前処理を使う。GVHMRは孤立frameで時系列を作り直すとモデルの実入力が変わるため、同一clipのbody view selectionが決めた連続segmentを使う。

| モデル | 採取位置 |
|---|---|
| DINO | `preprocess_frame` で短辺800・長辺上限1333・ImageNet正規化したテンソルを `_predict_compatible_batch` のstack直後で採取。元画像のH/Wもdecoder用に保存 |
| ViTPose-H | `pose_boxes` と `iter_frame_person_crops` で現行のbox geometryと正規化を使う。`_predict_crop_batch` で幅 `32:224` を切った `(B,3,256,192)` とbox xysを保持。flip test込みは `2B` のグラフ入力と反転復元を含む測定区間を記録 |
| HMR2 | `iter_person_crops` または `iter_frame_person_crops` で作成した正規化済みcropを `Hmr2FeatureExtractor._predict_crop_batch` の `img` で採取。boxとframeを保存 |
| GVHMR | `GvhmrMeshRecovery._predict_impl` で構築するネットワーク入力を、`body_view_selection`・`hmr2_features` から固定。2D関節／confidence、box、HMR2画像特徴、カメラ内パラメータと角速度、length、mask等の全入力を同じsegmentに紐づける |

各モデルの初回は B=1（flip併用ViTPoseの内部B=2）で比較する。同じ入力値・shapeで5条件が通る組を1行とする。比較に可変shapeが必要なら実入力の長さ・H/Wを先に列挙し、shape bucketごとに時刻・VRAM・精度を分ける。TensorRTのdynamic profileには各テンソルのmin/opt/maxと対応する長さ・maskを記録し、測定中はshapeを一定にする。範囲外shapeの拒否も確認する。境界shapeは先に個別試行し、16 GiBホストで同じjob内に複数エンジンや大きなactivationの同時常駐を避ける。

## 速度とVRAM

- 共有training queue `--resource all`。条件ごとに新規プロセスでロードし、変換・校正・ビルドは別プロセスで終えてから測る。PyTorchのロード時VRAMピークを含め、最初のCUDA context初期化後にallocator統計をリセットする。エンジンビルド時のメモリは別扱い。
- 各条件で30回warm-up。固定入力順を同じように巡回し、後続300回を100回ずつ3ブロックにする。ホストからの転送を完了してから `perf_counter_ns` を開始し、推論をenqueueしてCUDA同期が戻った時点を終端にする。H2D、D2H、decode、前・後処理、ロード、プロファイラを除外する。GPUのkernel時間の合計は別欄。
- TensorRTの速度計測では実行contextの `nvtx_verbosity` を `NONE` に設定し、読戻し値を `measurement.json` に記録する。ビルドの `DETAILED` はinspector用で、詳細layer NVTXの生成コストが同期込みのlatencyに入るため実行時に無効化する。 [NVIDIAの実行context仕様](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/ExecutionContext.html)
- 時刻の中央値・P95は線形補間の標本分位点。IQR（P75−P25）とMAD（中央値からの絶対偏差の中央値）およびブロック別値を残す。throughputは `全推論件数 / 同期済み時刻の総和`。CLIPはimages/s、GVHMRはsequences/sとframes/sをshape別に記録する。
- `torch.cuda.max_memory_allocated`／`max_memory_reserved` はPyTorch allocatorの高水準。TensorRTの内部確保はNVML側で観測する。NVMLはデバイスのused bytesをCUDA context前、ロード後、warm-up後、3ブロックの各同期後、精度取得後にサンプルする。最大値とcontext前との差を記録する。Windows側の消費も含むサンプル最大であり、瞬間的な上限の捕捉やプロセス別の厳密差分は保証しない。時刻計測中の連続照会は避ける。

## GPU内部指標

値の取得は時刻計測とは別プロセス。同じ30 warm-upと入力転送を済ませ、固定の代表入力だけを `cudaProfilerStart/Stop` と `steady_inference` NVTXで囲う。CLIPは計測値を見る前に `clip_000/cam0/frame 178/player 0` を代表にした。ViTPose等の代表も結果を見る前に、固定入力一覧と代表shapeから決める。

- **nsys** 2026.5.2の `--trace=cuda-sw,nvtx` を使用。SQLiteの `CUPTI_ACTIVITY_KIND_KERNEL` から同期を含むNVTX範囲内のGPU activityを1行1件として数える。CUDA API呼び出し、memcpy、warm-upは除く。`sum_kernel_duration_ns` は並列実行区間も各カーネルの時間を足す。
- **ncu** GB206（CC 12.0）で `dram__bytes_op_read.sum`、`dram__bytes_op_write.sum` を各kernelから合算する。`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` は各kernelの `gpu__time_duration.sum` で加重する。memory stallは `smsp__warp_issue_stalled_{long_scoreboard,short_scoreboard,lg_throttle,mio_throttle}_per_warp_active.pct` を理由別に同じ重みで記録する。long scoreboardはL1TEX依存、short scoreboardは主にMIO依存を示す。スロットルの2指標は各パイプの命令待ち。これらはkernelごとのstall率を時間で要約した値で、GPUが停止した壁時計時間の割合は表さない。
- TensorRTの `ncu` 収集では実行contextの `nvtx_verbosity` を `NONE` にする。Nsight Compute 2025.3.1は詳細NVTXラベルのJSON内のカンマでraw CSVの列をずらすため、この設定で外側の `steady_inference` の範囲を収集する。同じplanと代表入力を使い、各条件で `nsys` の範囲内kernel数と照合する。`--profiles ncu --profile-from` の追測は元の `result.json` にある取得済み `nsys` の件数を使う。件数に差があれば `coverage_mismatch` を記録し、非ゼロで終了する。
- `--ncu-timeout` はカウンタ取得の制限秒数を指定する。時間切れのCSVから得た途中の行数は診断に使い、指標は全カーネルを取得できたときに記録する。
- `ncu` のkernel replayは計測やキャッシュに影響し得るため、独立したlatencyの値と区別する。cacheとclock controlはnone。raw CSVの欠測と権限エラーは理由を残す。別GPUでは `ncu --query-metrics` で名前と単位を照合してから、同じ意味を持つ値として比較する。
- 律速判断はDRAM bytesとkernel時間、warp stallの内訳、Tensor pipe、nsysのkernel数を合わせて行う。単一の比率から結論を出さない。層precisionはengine inspectorとtraceのkernel名も照合する。

## 精度の指標と許容差草案

条件0と同一入力で、欠測・NaN・ID変更・不一致母数を先に集計する。共通有効集合では各誤差の中央値・P95・最大と、clip/camera単位の外れを残す。以下は人間が Phase 2 採用を判断するための暫定値。下流E2Eの合否は別途評価する。

| モデル | 条件0との比較 | 許容差の草案 |
|---|---|---|
| DINO | 現行のconfidence閾値でdecode後、画像ごとにIoU ≥ 0.50のboxだけで一対一割当。対応数を最大化し、同数では総IoUを最大化する。閾値未満と未割当は参照側の欠落・候補側の追加に数える。IoU、xyxy px差、confidence差の分布は対応成立ペアだけで集計する | IoU P05 ≥ 0.98、score絶対差P95 ≤ 0.01、欠落は参照総数の1%以下、追加は候補総数の1%以下 |
| ViTPose-H | 同じbox・flip設定・UDP decoderで、17点の元画像xyのL2距離とconfidence差。joint有効maskを照合 | xy P95 ≤ 1 pxかつ最大 ≤ 5 px、confidence絶対差P95 ≤ 0.02 |
| CLIP-ReID | 1280-dのL2正規化特徴の対応crop cosine。clip内で各カメラ・playerの最初の有効cropをgalleryにし、queryとは異なるカメラのgalleryだけで最近傍IDを出す。条件ごとにgallery/queryを算出し、同じ順序で同点を解決 | cosine P05 ≥ 0.995、FP32とのID一致 ≥ 99%。変化したqueryとgalleryの候補差を精査 |
| HMR2 | `(F,1024)` 画像特徴に対するper-crop cosineと相対L2誤差、各次元の絶対差。対応するframeと欠測を照合 | cosine P05 ≥ 0.995、相対L2 P95 ≤ 0.05 |
| GVHMR | `body_pose/global_orient` をSO(3)角度差に変換。 `betas/transl` の絶対差と、同じSMPLモデル・座標系での関節L2距離を比較。lengthとmaskも照合 | 回転角P95 ≤ 1°、joint距離P95 ≤ 10 mm、maskと有効frame数一致。 `betas/transl` は差分を提示して下流3Dと併せて判断 |

DINOの対応付けは `dino_accuracy.py` の `compare_dino_frame` を使う。対応が0件の場合、対応ペアの分位点は欠測として扱う。IoU 0.50はPhase 1の暫定値で、Phase 2で人間が許容差と併せて見直せる。

[Nsight SystemsのCUDA trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace)・[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)・[ModelOpt ONNX PTQ](https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html)
