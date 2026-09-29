# 3D Ball Refiner (#936)

CPUでの入力分布・合成系列生成と、絶対x0を予測するdiffusionの土台を提供する。
生成と方式比較の結果はknowledgeに記録する。pipeline接続は未実装。
要件の正本は [#936](https://github.com/Motoki0705/tennis-lab/issues/936)。
2D契約は [親README](../README.md#2dモデルのapi) を参照する。

## 今回の比較

A（成分組合せのMahalanobis最適化＋Laplace）、B（適応voxel積分）、
C（2D分布の標本化＋三角測量）を同じ合成入力で比較する。
3D密度の単位は m⁻³、NLLは自然対数、coverageは混合全体のHDRで測る。
点推定の平均誤差だけで方式を選ばない。raw detectorは入力にしない。

位置prior・presence周辺化・Laplace evidenceの定義と限界は
[共有geometry API](../../../utils/geometry/probabilistic_triangulation/README.md)を正本とする。
`triangulation.frame_observations()` は既存の `pixel_moments()` を使って、
B軸を呼び出し側が同期させたcamera順のV軸へ写す。成分は平均しない。

比較用cameraは
[`fixtures/meiji_video_002_clip_010.json`](fixtures/meiji_video_002_clip_010.json) の
video_002/clip_010、cam0/1/2（1920×1080）。
既存court校正artifactのパスとSHA256を含み、実ボールの観測・評価には使っていない。
比較のCLIは `python -m src.tasks.ball_refiner.scripts.compare_triangulation`。
`--fixture` と未使用の `--output` を絶対pathで明示し、CPU/native threadを1に制限する。

## 59.94fps合成系列の生成

設定の正本は [dataset_plan.yaml](dataset_plan.yaml)。
`synthetic/` はBLCSの240Hz原系列を正確な60000/1001Hzへ線形補間し、
合成3D → source画素 → `BallGMM2D` → `pixel_moments()` → 明示的なA/B併用の順で生成する。
全camera集合と全共分散を保存する。v2は保存済み#935 pilotのK=4を保ち、
全125成分を列挙する。点推定への置換はしない。

`timebase.py` は実際に残ったshot区間のイベントだけを採用し、native frame/秒と
最近傍frameを保持する。打球・bounceの±5frameとnet通過付近を物理lossから除外する。
fence時刻を返さないsimulatorに対しては、fence近傍も保守的に除外する。
`observations.py` は#934の録画splitを維持し、各sceneの摂動cameraを真/推定の双方に
使うclean geometry条件を作る。真/推定行列は別fieldで保存する。
校正誤差条件は未実装。実ボール座標や注釈は入力にしない。

`calibration.py`は#959の較正用validationに保存済みの全GMMと教師から、camera別・
観測/人工証拠欠損別にuv残差、全Cholesky因子、全混合logit、存在logitを抽出する。
checkpoint・manifest・全NPZのSHAと誤差/共分散比・重み・正例の存在統計は
[暫定較正bundle](../../../../knowledge/runs/run-i936-provisional-degradation-r5-s936/calibration.json)を正本とする。
GPU再推論や実3D軌道評価は行わない。これは**文脈なし旧detector pilotからの暫定劣化**であり、
新detector/person contextの最終較正ではない。

`observations.py`はcameraごとに連続する採点frameを最大16frameのblockで再標本化し、
全成分のuv残差を合成投影へ移す。欠損中もamodal存在logitを保持し、visibilityから
不在を作らない。frame間の相関はblock内だけ、camera間は独立という暫定近似。
32/64frameのgapは保存済み1/4/8/16frame gapからの外挿である。画面外の存在logitは
負例不足のため設定で明示した仮定。平均は#935 headと同じ[0,1]へclipし件数を保存する。
再標本化元の全row indexとbank hashを各ラリーに保存し、readerで全重み・共分散・存在を照合する。

積分は[共通の収束判定](../../../utils/geometry/probabilistic_triangulation/README.md#積分の収束判定)を使う。
frame/成分別の収束flag・達成差分・使用予算・全履歴を保存する。上限で未収束のframeも
最後の全分布を保持し、収束済みへ読み替えたり学習loaderで黙って除外したりしない。
数値失敗は別の明示的errorである。v1の仮定劣化/K=3からのデータ移行は再生成で行う。
過去のv1はschemaを指定した読込だけを維持し、新しい生成には使用しない。

BLCSが既知prefixで棄却した物理提案だけを設定の有限予算で再標本化し、
全提案seed・棄却理由・採用seedをmetadataへ残す。元のnative上限でsimulateした後に
保存prefixを切り出す。未知例外、予算枯渇、短すぎるrally、solver失敗、非SPDは停止する。
別seedでの三角測量再試行、成分削除、jitter、自動resumeはしない。
1ラリーが失敗しても提出済み全ラリーの成否をmanifestへ集め、datasetはfailedにする。

生成CLIは `python -m src.tasks.ball_refiner.scripts.generate_synthetic_3d`。
`--project-root`（作業checkout）、`--data-root`（共有data）、
`--plan`（上記YAML）、`--output`（新しいDATA内directory）を絶対pathで指定し、
`--mode smoke`、`--mode dev`（64/16/16ラリー）、または `--mode pilot` を明示する。
OMP/MKL/OPENBLASのthread数を1にして起動する。process数はYAMLの値で最大4。

各rallyを可変長のNPZ+JSONで保存し、進捗manifestと16frame間隔の各rally progressはatomicに更新する。
timestamp/cameraはfloat64、軌道/GMMはfloat32、maskはbool。
float32 export後もSPDと有限性を検査する。未完了/失敗は`complete`にしない。
入力設定・校正・生成codeのSHA、全イベント、実frame/bytes、simulation/triangulation時間、
process RSS、全量生成の線形予測をmanifestへ記録する。出力は上書きしない。
RGB生成、実Meiji評価、pipeline統合はこの入口の範囲外。

## Diffusion scaffold

`diffusion/model.py` の `TrajectoryDenoiser` は、各3D成分の平均・全共分散・
camera subsetを非線形に符号化してから混合重みで集約し、時間Transformerへ渡す。
出力headは正規化court座標の絶対位置x0とhit/bounceの2 logits。
座標は共有の `isotropic_half_length` 契約を使い、三角測量の平均への残差加算はしない。
実frameの欠損と右paddingは別で、全camera欠損でも実frameはattentionへ残る。

`diffusion/flow.py` は `x_t=(1-t)noise+t*x0` の経路でx0を回帰し、
`v_t=(predicted_x0-x_t)/(1-t)` のEuler法でsampleする。
t=1で速度を評価しない。x0 MSEは一様tで学習し、velocity MSEで見れば
`(1-t)^2` の重みに相当する。sample全体の平均・不偏共分散をuncertaintyとして返す。
同じbackboneの1-step回帰はnoisy-state/time入力を0に固定する。

`diffusion/losses.py` はx0、全2D成分を使うStudent-t再投影、自由飛行の重力残差、
hit/bounce BCEを実装する。再投影はbehind predictionを捨てずdepth penaltyを付ける。
重力項はdrag/Magnus/windを再現しない**弱いprior**であり、BLCSの完全な物理残差ではない。
イベント前後のmaskと差分stencilの全3frameが有効な箇所だけに適用する。
現段階では実datasetの学習loader、品質評価、deploymentを提供しない。

## CPU/GPU memory diagnostic

数値の正本は [memory_smoke.yaml](memory_smoke.yaml)。
`python -m src.tasks.ball_refiner.scripts.memory_smoke_3d` に
`--config`、`--fixture`（camera-only JSON）、新しい `--output` を絶対pathで指定し、
`--device cpu` または `--device cuda` を必ず明示する。CUDAは共有training queue専用。

入力は `analytic_memory_fixture_v1` と明示した解析的tensorで、失敗datasetの代用品を
本学習へ流す機能ではない。全64成分・相関2D分布・64frameの分散拡大を持つ。
このfixtureでは計算graph/100 updates/VRAMを独立に測定するが、
データ経路の完走、物理精度、汎化、較正の証拠にはならない。
出力checkpointは `diagnostic_only=true` で、学習pilotへ再利用しない。

runnerはfp32/eager/100 updates、明示allocator上限と時間上限で実行し、
NaN/Inf、予算超過、CUDAなしをerrorにする。別deviceや小batchへの自動切替はしない。
各updateの全loss/gradient normをJSONLへ即時保存し、最終manifestに時刻・
peak allocated/reserved bytes・checkpoint SHAを記録する。
GPUのcontext/library分はPyTorch allocator測定に含まれない。

本学習・同backbone回帰・合成評価・Meiji LOCOは、開発datasetと最終的な#935の
劣化較正を経て別runで実施する。現在の方式比較とmemory smokeを性能の採否に使わない。
