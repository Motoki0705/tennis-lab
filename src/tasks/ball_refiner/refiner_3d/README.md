# 3D Ball Refiner (#936)

CPUでの入力分布・合成系列生成と、絶対x0を予測するdiffusionの土台を提供する。
生成と方式比較の結果はknowledgeに記録する。pipeline接続は未実装。
要件の正本は [#936](https://github.com/Motoki0705/tennis-lab/issues/936)。
2D契約は [親README](../README.md#2dモデルのapi) を参照する。

## 今回の比較

A（成分組合せのMahalanobis最適化＋Laplace）、B（適応voxel積分）、
C（2D分布の標本化＋三角測量）を同じ合成入力で比較する。
K=4の固定標本ではH（固定voxel予算のhybrid）、ray、adaptive ray、固定20次rayも比較し、
既定のHを選んだ根拠を[knowledge](../../../../knowledge/nodes/ball_refiner_3d/000013-run-i936-k4-method-choice-r8-s93607.md)へ記録する。
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
合成3D → source画素 → `BallGMM2D` → `pixel_moments()` → 共通conditioning APIの順で生成する。
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

generatorの`--calibration-report <絶対project-path>/calibration.json`で、同じdirectoryの
`bank.npz`とreportを1入力として交換できる。bankのSHA、成分数、schemaを検証し、
展開済み設定・report SHAをmanifestへ固定する。未指定時はplanの明示したbundleを使う。
監査の再現CLIも同じ引数と`calibrated_distribution()`を使う。将来の較正比較では
固定rally/frame/maskとrally seedを使ってbootstrapし直すため、比較する両方のreportを
明示する。元の保存済み2D入力を再積分する監査とは別条件として記録する。

`observations.py`はcameraごとに連続する採点frameを最大16frameのblockで再標本化し、
全成分のuv残差を合成投影へ移す。欠損中もamodal存在logitを保持し、visibilityから
不在を作らない。frame間の相関はblock内だけ、camera間は独立という暫定近似。
32/64frameのgapは保存済み1/4/8/16frame gapからの外挿である。画面外の存在logitは
負例不足のため設定で明示した仮定。平均は#935 headと同じ[0,1]へclipし件数を保存する。
再標本化元の全row indexとbank hashを各ラリーに保存し、readerで全重み・共分散・存在を照合する。

既定は[固定予算H](../../../utils/geometry/probabilistic_triangulation/README.md#固定予算のconditioning)。
設定キー`boundary_convergence`は既存schemaを維持し、方式と固定予算を格納する。
`integration_convergence_assessed`を保存し、readerは未評価を収束済みとする改変を拒否する。
収束を測る方式を明示した場合はframe/成分別のflag・達成差分・使用予算・全履歴も保存する。
未評価/未収束でも最後の全分布を保持し、学習loaderで黙って除外しない。
数値失敗は別の明示的errorである。v1の仮定劣化/K=3からのデータ移行は再生成で行う。
適応rayの補助誤差とchart選択もNPZへ保存する。`integration_component_metric_codes`は
0=適応chart対象外、1=局所Hessian、2=白色化画素/log-depth単位軸。readerは方式codeとの一致を検証する。
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
最終的な対照評価・実Meiji評価・deploymentは未完了。

## 保存済みsmokeでのCPU学習loop

`diffusion/data.py`は検証済みラリーから全3D/2D成分を保持したwindowを作り、
#935の`pixel_moments()`でsource画素へ変換する。右padding、実frameの欠損、
積分の未収束を区別する。未収束frameを省略せず、収束flagのない歴史的datasetは
先に再積分を要求する。flag付きdataを使うplumbingは設定で明示的に許可する。

設定の正本は[training_smoke.yaml](training_smoke.yaml)。
`python -m src.tasks.ball_refiner.scripts.training_smoke_3d`に
`--dataset`、`--config`、新しい`--output`を絶対pathで渡す。
CPU/native thread1のみ。12ラリーの全windowでforward/lossを確認し、
ソート順で最初の2つのtrainラリーのprefixだけをtiny overfitする。
val/testはplumbing確認だけで、更新・checkpoint選択・品質の結論に使わない。

毎updateで新しいflow time/noiseを使い、x0/robust再投影/イベントmask付き弱い重力残差/
イベント分類を同時に学習する。固定noise/timeのloss probeと、固定noiseからの8-step生成を
前後で測定する。全loss曲線、入力SHA/window identity、未収束数、形状、checkpointを保存。
checkpointは`diagnostic_only`であり、本学習・汎化性能・diffusionの対照に対する優位の証拠ではない。

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

## H dev setの検証と短時間学習

`python -m src.tasks.ball_refiner.scripts.verify_synthetic_3d` は、絶対pathの
`--dataset` と新しい `--output` を受け取る。96件をreaderへ戻し、保存dtype/SPD、
125成分、presence質量、seed/geometry/split/入力hash、イベント・欠損・収束診断を検証する。
train/valの全画面内camera-frameだけを#935共通の全混合HDRで採点し、
observed/人工gap別の50/90/95%被覆と分母を保存する。testは保存契約の検証だけに使う。

設定の正本は [training_dev.yaml](training_dev.yaml)。
`python -m src.tasks.ball_refiner.scripts.training_dev_3d` に絶対pathの
`--dataset`、`--config`、新しい`--output` と、明示的な `--device cuda` を渡す。
CUDAは共有queue専用。CPU指定は通常検証用で、自動device切替はしない。
固定Hの全入力を保持し、未評価/未収束flagは診断として数える。trainの全ラリーを
窓にし、同じ初期重み・窓のshuffle順からflowと1-step回帰を別々に学習する。
validationの文脈は設定で必ず明示する。旧対照の `validation_frames: null` は全ラリー、
`validation_frames: 128` は学習と同じT128/stride128で推論する。
testラリーを開かず、checkpoint選択や調整をしない。

各更新の4損失・速度・メモリをJSONLに即時保存し、明示した全評価時点のval予測・曲線PNG・
開発専用checkpointを残す。RMSEは全体、全camera人工gap、全camera証拠なし、hit/bounce±5。
加速度/jerkは実秒の2/3階差分で、全区間と全stencilが自由飛行の区間を別集計する。
再投影は真camera上の画面内合成GTへの画素誤差で、behind-camera件数を別報告する。
有限な再投影値は正depthに条件付きなので、不正depthが残る結果を改善とは扱わない。
flowは平均軌道と全サンプルの指標を併記し、平均化だけでjitterを隠さない。

### 同一valの単純ベースラインと更新数延長

`baselines.py`は保存された全3D混合の平均、最大重み成分平均、混合平均への
#929重力RTSを無学習・無調整で比較する。これは点推定ベースラインであり、
生成器やモデルへ渡す全成分の条件分布を変更しない。RTSの数値処理・イベント検出は
`baseline_rts.py`に記載したcommitの抽出で、教師イベントを使わない。
全frameに分布があるため欠損中も点推定を処理する。#929 pipelineのbounds/速度gateは
ここでは件数を記録する診断とし、ラリーを除外せず採点する。数値失敗は停止する。

`python -m src.tasks.ball_refiner.scripts.compare_dev_baselines_3d`はCPU専用で、
絶対pathの`--dataset`、旧2k runの`--training-output`、新規`--output`を受け取る。
dataset/valラリーのhashと予測に保存したGT・mask・cameraを照合し、
旧指標の再計算一致を要求する。train/test NPZは開かない。
比較表・全指標JSON・baseline予測を保存する。

学習runnerも同じベースラインを初めに保存する。`evaluate_updates`は初期0から
最終更新までの明示的な昇順リストで、各時点の予測は
`<arm>/predictions/update-<番号>/`に保存する。
[training_dev_long.yaml](training_dev_long.yaml)は更新数、評価時点、時間予算だけを
短時間設定から変えたdev実験で、初期化・窓順・モデル・loss・較正は同じ。
[training_dev_anchored_t128.yaml](training_dev_anchored_t128.yaml)は20k設定のvalidationを
学習と揃え、評価時点を0/2k/5k/10k/20kに固定する。新bankのdevを明示して実行する。
窓推論は`context_inference.py`をCPU診断と共有し、絶対時刻・右padding・
短い末尾の重複は早い窓の採用を維持する。初期noiseと教師ありprobeは
全ラリーに一度だけ生成して窓へ切り出し、同deviceの評価時点間で固定する。
loss・加速度・jerkは全frameを一度ずつ継いだ元系列で計算し、窓境界も含める。
両armの完了時に全評価時点・平均/全sample・無学習baseline・GTを同じ
`comparison.json/md`へ出力し、GTと分母の一致を要求する。
可視camera数（occlusion/out_of_frameのどちらもない台数）で全指標も層別する。
差分は元時系列で計算し、加速度は中央frame、jerkは左中央frameの層へ割り当てる。
層ごとに離れたframeを連結して差分を取らない。

本学習・#929との最終対照・Meiji LOCOは最終的な#935較正を経て別runで実施する。
開発用の短時間比較を、最終精度・パレート優位・本番採用の証拠にはしない。

### 条件tokenのCPU read-out診断

`python -m src.tasks.ball_refiner.scripts.probe_conditioning_3d`に絶対pathの
`--dataset`、完了dev runの`--training-output`、新規`--output`を渡す。
最終flow checkpointを凍結し、実モデルの`encode_condition()`を共有して、
全成分の非線形符号化・重み付きpool直後のtokenから全混合平均を読み出す。
全train実frameで切片付き線形headをfloat64 SVD最小二乗でfitし、全valで固定評価する。
正則化・seed/checkpoint選択・test NPZ読込は行わない。合成GTをfit教師に使わない。
rankと特異値を報告し、rank不足を別solverやridgeで黙って修正しない。

正規化round trip、raw特徴からの平均復元、可視camera数別誤差、全入力/重みhash、
head係数と予測を保存する。診断基準と解釈は実験knowledgeを参照。
良いread-outはtokenに平均位置が残る証拠だが、時間Transformerがそれを利用できる保証ではない。
悪い線形read-outだけで非線形な情報復元も不可能とは断定しない。
CPU1 thread専用で、モデル本体・生成器・損失・pipelineは変更しない。

### T128文脈のCPU診断

`python -m src.tasks.ball_refiner.scripts.probe_context_3d`に、上記read-outと同じ
`--dataset`、`--training-output`、新規`--output`を指定する。
20kの両armを凍結し、全16 valを全ラリー/T128でCPU推論する。
全ラリーで生成した初期noise・教師ありprobeのtime/noiseを両文脈で共有する。
T128/stride128、絶対時刻、右paddingを保ち、重複する短い末尾は先の窓を採用する。
全frameを一度ずつ採点し、loss/加速度/jerkは継ぎ目を含む元時系列で計算する。
train/testは開かず、判定基準とGPUとの差の扱いは実験knowledgeに記録する。

### Residual bank交換時の3D条件監査

`python -m src.tasks.ball_refiner.scripts.audit_conditions_3d --dataset <絶対path>
--output <新規絶対path> --samples <各MC標本数>`で、全96件のmanifest/JSON/hashを監査し、
train/valだけをreaderへ戻して全3D混合のGT NLL・HDR50/90/95被覆/体積・混合平均RMSEを測る。
可視camera数で層別し、16 valの混合平均と#929 RTSも固定設定で比較する。
`--audit-only`は品質採点を省いた事前監査。いずれもtest NPZはhash確認だけで配列を開かない。
HDR体積はR³上の独立MC推定、報告するMC標準誤差は推定閾値に条件付き。
`compare_condition_reports()`はbank以外のplan・全96件のseed/metadata・
80件の軌道/camera/イベント/gap maskのhash一致を要求する。
