---
id: run-ball3d-topk-multihypothesis-20261103
type: run
task: tennis_scene
sequence: 34
recorded_at: '2026-11-03'
title: clip_000でtop-K多仮説対応付けをCPU replayしLOVO悪化とcoverage低下を確認
issue: 1560
provider: codex
date: '2026-11-03'
status: done
config:
  clip_id: meiji_3cam/video_000/clip_000
  num_frames: 1010
  tuning_frames: [0, 505]
  primary_evaluation_frames: [505, 1010]
  parameter_selection: predeclared_defaults_without_label_tuning
  top_k: 8
  max_hypotheses: 24
  beam_width: 96
  max_missing_frames: 3
  gpu_used: false
  inference_rerun: false
metrics:
  baseline_primary_lovo_median_px: 9.340188442451916
  method_primary_lovo_median_px: 10.7804879790132
  baseline_primary_lovo_p90_px: 27.078044438775773
  method_primary_lovo_p90_px: 103.59736745483048
  baseline_primary_lovo_count: 825
  method_primary_lovo_count: 533
  primary_lovo_eligible: 1462
  baseline_primary_lovo_coverage: 0.5642954856361149
  method_primary_lovo_coverage: 0.3645690834473324
  paired_two_method_primary_count: 353
  paired_baseline_primary_median_px: 8.77846832956076
  paired_method_primary_median_px: 9.27354366119481
  tests_passed: 106
artifacts:
  run_dir: knowledge/runs/run-ball3d-topk-multihypothesis-20261103
  output_dir: outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/topk_multihypothesis_v2
  reproduction: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/repro.sh
  metrics: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/metrics.json
  folds: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/fold-metrics.csv
  tests: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/test-summary.json
  inventory_validation: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/validation.json
  ci_fix_validation: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/ci-fix-summary.json
  composition_validation: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/composition-summary.json
  cohort_metrics: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/cohort-metrics.json
  cohort_artifacts: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/cohort-artifacts.json
parents:
- run-ball3d-foundation-review-20261103
relations: []
papers: []
tags: []
repro:
  commit: 49cc2b31f9e3dee3de28dc4d032d17bfed7199de
  command: bash knowledge/runs/run-ball3d-topk-multihypothesis-20261103/repro.sh BUNDLE_DIR VIDEOS_DIR GROUND_TRUTH_JSON PRIOR_BASELINE_DIR NEW_OUTPUT_DIR
---

## 比較の条件と結果

親 #1554 revision 6、子 #1560。公開clip_000の3camera・1010frame、固定bundleだけをCPU replayした。
基盤のloader・runner・保存/再読込・採点・動画/図/crop/HTMLを使い、独自loaderやscorerは作っていない。
GTは全予測の保存後に採点側だけで読み込む。直接観測2589点・承認推定370点を採点し、
cam0末尾外挿28点と座標なし43点を補完しない。観測/推定・イベントの区別は共通scoresに残る。
方式設定はラベルを読む前にcommitし、`frozen-config.json`へ保存した。前半を含めラベルでの調整は行っていない。
主評価は[505,1010)、前半[0,505)と全区間は併記する。後半を見て方式設定を変えていない。

| 区間 | 方式 | LOVO 中央値/p90 px | >20px率 | 採点数/対象数・coverage | 全view有効3D率 |
|---|---|---:|---:|---:|---:|
| **主評価 [505,1010)** | baseline | 9.340 / 27.078 | 16.36% | 825/1462・56.43% | 83.17% |
| 同 | top-K多仮説 | 10.780 / 103.597 | 28.52% | 533/1462・36.46% | 41.39% |
| 前半 [0,505) | baseline | 10.716 / 28.483 | 16.44% | 821/1497・54.84% | 86.34% |
| 同 | top-K多仮説 | 11.903 / 84.848 | 28.86% | 589/1497・39.35% | 43.37% |
| 参考 [0,1010) | baseline | 10.208 / 28.196 | 16.40% | 1646/2959・55.63% | 84.75% |
| 同 | top-K多仮説 | 11.361 / 98.961 | 28.70% | 1122/2959・37.92% | 42.38% |

主評価では中央値+1.440px、p90+76.519px、>20px率+12.15pp、採点数-292点・coverage-19.97ppで悪化した。
各camera除外・全view・区間別の値は [fold-metrics.csv](../../runs/run-ball3d-topk-multihypothesis-20261103/fold-metrics.csv)。
主評価のcam0/1/2除外はtop-Kでそれぞれ8.949/33.509px（166/473）、14.315/72.753px（151/484）、
10.792/279.974px（216/505）。baselineは7.920/14.235px（271/473）、11.990/25.147px（231/484）、
9.262/40.718px（323/505）であり、特にcam2を除いた2viewの裾が大きい。

## 共通点と欠測を併せた判断

2方式の共通LOVO353点でもbaseline 8.778/27.825px・>20px率15.01%に対し、方式は9.274/34.073px・17.85%で悪化。
oracleを含む3方式共通348点でもbaseline 8.669/25.190px、方式9.101/29.883px、oracle 7.135/18.918px。
共通集合も可用性を条件とする23.8%の支持に限られ、欠測を回復した比較ではない。
LOVO集約は個々の残差のpoolで、camera別中央値の平均ではない。

主評価全viewでは505frameすべてに幾何仮説があったが、動的計画法の出力は209frameで296frameを欠測にした。
選択した候補517件のうちrank1が475件、rank2以降は42件。8回の履歴リセット/再捕捉があった。
[candidate-diagnostics.json](../../runs/run-ball3d-topk-multihypothesis-20261103/candidate-diagnostics.json) はラベルを使わない診断。
運動の外れが少ないことを精度改善とみなさない。欠測を増やしながらLOVO裾も悪化しているので、今回方式の採用は勧めない。
採否は人間の統合比較後の判断に残し、既定pipelineは維持する。

## 方式と先行知見

方式・単位・出力契約の正本は [ball_association README](../../../src/tennis_scene/ball_association/README.ja.md)。
gate前のvalidなtop-8をscoreで切らず、全pair候補のDLTと他viewへの幾何/score拡張から仮説を作る。
pair-onlyも含み、幾何棄却後の上位24仮説を96状態の二次運動beam DPで選ぶ。
速度上限65m/s、加速度コストは上限で打ち切り、bounce/hitを有限コストにする。実イベント/GTは入力にしない。
3frameを超える欠測で履歴をリセットし、欠測自体は補間しない。厳密な全組合せViterbiではない。

[#1558の診断（PR #1670）](https://github.com/Motoki0705/tennis-lab/pull/1670) とそのknowledgeを読んだ。
前半argmax失敗はcam0が102点（top-8で63点回収可能）、cam2が125点（39点）、cam1が44点（29点）。
mask前で候補を扱う動機になる一方、cameraごとに回収可能性が違い、top-8に正解がない点も多い。
同診断の「既存採用maskを保つcandidate oracleの改善は小さい」「LOESS除去は共通点の精度を悪化」も踏まえた。
今回は候補＋時間の選択だけを検証し、選択後LOESSは追加していない。
候補剪定・未較正score・nullと加速度コストの釣合い・2D LOESS欠如が限界の候補だが、
独立ablationを行っておらず、各要因へ悪化量を割り当てることはできない。

## 検証・再現と制限

実装前に [test-plan.md](../../runs/run-ball3d-topk-multihypothesis-20261103/test-plan.md) を作成した。
非対称camera/frame、途中の順位逆転と長い張り付き、欠測/再捕捉、急な方向転換、
既知の2m/0.3秒のgap速度、低scoreの保持、異なるK、設定拒否を軽いunitで検証する。
公開config→definition→assembler→方法→実codec、LOVO隔離とmainの固定期待配列をintegrationで確認する。
既定componentとmainの位置・mask・inliers・pixel・rejection配列は厳密比較する。
再投影配列だけは、後述のCI修正で検証したfloat32の丸め幅を許容する。
全106nodeのcollect-onlyと各phaseの実測はtest-collection/test-results/test-summary、正確なコマンドはtest.sh。
`-n 0 --collect-only` は106件/6.69秒、`--durations=0 --durations-min=0` は106 passed/29.13秒。
全19追加caseを含む各node/stageの厳密な時間はJSONに保持し、pytest全体の時間とcase合計を区別する。
加速度コストをゼロにする故障注入で数値assertが失敗し、元bytesへ復元後は成功した。既存nodeの削除・統合はない。

初回PRでは基盤#1669のmain merge後にorigin/mainを取り込み、実行コードと同じsrcでテストを再実行した。
その後の設定adapterの宣言修正と、実データへの影響確認は以下のCI修正欄に記す。
v1は初回run、v2は負の整数重み/bool設定の拒否を補強した後の同設定再実行で、
全8予測結果・全診断配列と比較MP4はbitwise一致。baselineの4foldはfoundation_v2ともbitwise一致した。
元store照合も有効856frameとmask一致、3D最大差は2µm許容内。GTや入力bundleは変更していない。

v2は約259.5秒、最大RSS1,133,200KiB。検出器再推論・GPU・学習・追加入力取得はない。
比較MP4は1200×1128・1010frameを全decode。代表0/252/505/757/1009、指標表、6パネルの軌道/分布図、
極端419/417/415/411/407/406の3camera cropを目視した。frame252で選択squareと候補順位の重なりを確認し、
frame757のunknown・frame1009の末尾外挿除外が正解markerとして描かれないことも確認した。
HTML参照切れ0、全129ファイルのサイズ/SHAと未宣言ファイルなしを照合した。記録はvalidation/verificationに保存。
大きな配列・動画・HTMLはoutputsに残し、knowledgeには小さな指標・図・実行証拠だけを保存する。

単一clip・固定の近似校正で、独立3D正解はない。oracleは数学的下限ではなく幾何参照。
beamとpair拡張は近似探索であり、完全なjoint候補列を保証しない。
次は同一条件の他方式との統合比較を行う。改良する場合も、未使用clip又は前半だけで
null/運動コスト、候補保持、選択後LOESSを独立に検証し、今回後半の再調整はしない。
学習ではないためTensorBoard・seed間学習曲線は適用しない。PRは人間の採否までmergeしない。

## PR #1672のCI修正と予測の再照合

旧head 87ed823のCI run 41603205543では、全10 CPUレーンが設定catalogの収集で失敗した。
`PipelineRuntimeConfig.ball_method` のPython既定値がcomposition-owned契約に違反し、
最初の違反を直すと `ball_topk` のPython既定値も同じ違反になるため、両方を必須fieldへ直した。
`from_config` の処理は旧headとAST一致し、YAMLも不変である。productionでは従来からstrict schemaで
全top-K fieldを必須としてcomposition値を渡している。新しいtestは0を含むCLI overrideと全fieldの到達を検証した。
ただし、この修正時点では`TopKConfig`と評価methodにPython既定値が残っていた。
監査が発見したadapter・schemaの確認だけでは不十分で、既定値のあるネスト型の探索漏れを見逃していた。
後述の追加修正で設定型・評価methodの既定値とfactoryを除き、対象symbolの発見を直接検証する。

もう1件は、main固定配列に対するreprojection_pxの厳密一致が機械依存の1～数ULP差を拒否したもの。
既定solverと入力組立5ファイルはmain記録とchecksumが完全一致する。
solverはfloat64のDLT/least_squaresのfitから残差を計算し、位置と残差をそれぞれfloat32へ格納する。
したがって、保存した位置の一致は元のfloat64 fitと残差の全bit一致を意味しない。
小fixtureでOpenBLAS kernelをCORE2へ切り替えると、位置・valid・inliers・UV・理由コードは完全一致のまま、
再投影7/36値だけが1 ULP（最大1.4551915228366852e-11 px）変わった。
NumPyのAVX2/FMA3/AVX512F無効化だけでは差は出ず、今回のprobeではBLAS経路の違いで再現した。
保存済みfloat32位置から残差を再計算すると約1.1533e-6pxの別の差になるため、その再計算へは変更しない。

testはreprojection_pxだけ `rtol=4*float32.eps=4.76837158203125e-7, atol=0` に変更した。
CIの最大相対差1.9e-7を含む丸め幅で、20pxでの許容差は9.5367431640625e-6px。
他の配列の比較は厳密一致を維持し、ゼロ残差もatol=0で厳密に保つ。pixel単位の方式比較の差を隠す許容幅ではない。
probeは [reprojection_probe.py](../../runs/run-ball3d-topk-multihypothesis-20261103/reprojection_probe.py)、
実測は [ci-fix-reprojection-probe.json](../../runs/run-ball3d-topk-multihypothesis-20261103/ci-fix-reprojection-probe.json)。

修正commit 810d833で、修正後のHydra composition値とv2のfrozen configの全field一致を確認し、
固定clip_000を基盤runnerで全view/LOVOへCPU再実行した。baselineと本方式の計8結果で、
位置・valid・used-view mask・frame軸・全診断配列がv2とbitwise一致した。
v2の全129ファイルのサイズとchecksumも再照合した。採点関数・camera・参照も変更しておらず、
再計算した予測が同じなので、上の誤差・coverageおよび悪化の結論は変わらない。
v2 directoryは不変。照合runは `outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/topk_multihypothesis_ci_fix_v1/`、
小さな証拠は [ci-fix-prediction-check.json](../../runs/run-ball3d-topk-multihypothesis-20261103/ci-fix-prediction-check.json)。
約60.6秒、最大RSS635,036KiB、GPU/再推論/動画生成/GTの方式入力への利用はない。

事前設計は [ci-fix-test-plan.md](../../runs/run-ball3d-topk-multihypothesis-20261103/ci-fix-test-plan.md)。
設定監査全体・synthetic visualization契約・設定/結合テストは427 passed、既存の空parametrizeによる1 skip、381.33秒。
関連suiteとDINO回帰は135 passed/107.85秒、collect-onlyは135件/96.27秒（すべて -n 0）。
両groupの重複を除いて528成功node。phase時間はci-fix-audit-tests/ci-fix-related-tests、要約はci-fix-summaryに記録する。
既存のcold import2caseは123.87/114.75秒、全source監査は40.35秒で5秒目安を超える。
新しいprocessでのimport順と全source/catalogの契約を検証する既存nodeであり、mockや対象削減で代替しない。
追加したcomposition到達テストと関連135caseはすべて5秒以内、最大setupも1秒以内だった。
DINOの指定nodeは旧CI・ローカルでは成功したが、修正head 5c3627c3dのCI run 41654669072・CPUレーン9で
`DINO functions already reference a different extension` が発生した。親の事前指示に従い、
a8982a7e0の `tests/unit/submodules/models/dino/test_extension.py` の変更だけを同内容で取り込んだ。
fake packageとともに先にimportされた子moduleのMSDA参照も退避・復元するテスト分離修正で、
productionコードの変更はない。既存cacheなし/ありを含む同ファイルを再実行し、そのreceiptをci-fix-dino-testsへ保存した。
実データを再照合したcommitからの追加変更はこのtestと記録だけで、v2不変の結論を維持する。
push後の対象headの全CI結果はPR #1672へ記録し、人間の採否までmergeしない。

## TopK設定のcomposition所有権の追加修正

2026-11-04、親の追加指摘で、前回修正head 2adc9293bの`TopKConfig`に実際には16項目のPython既定値が
残っていたことを確認した。評価adapterの`settings`にも`default_factory=TopKConfig`があり、
`min_frames=5`もPythonで補完していた。これではpipeline YAMLの変更が比較runへ反映されない。
この時点でCI成功を設定型全体の既定値監査の証拠とした判断は不十分だった。

`TopKConfig`を全項目必須にし、`TopKMultiHypothesis`もsettings/min_framesを必須にした。
coreの`min_frames`とcomponentの`enabled`の省略値も除いた。標準pipelineは従来からこれらを明示している。
schemaは設定instanceを生成せず型注釈から作る。評価run.pyはHydraで標準pipelineをcomposeし、
`PipelineRuntimeConfig.from_config(..., bind_inputs=False)`の同じ設定を明示して渡す。
単体testもmodule-scopeの実compositionを1回使い、結合testもproductionで得た設定を評価adapterへ渡す。
今回追加した設定型・factoryを確認し、`TopKConfig`・評価adapter・`PipelineRuntimeConfig`の全fieldで
default/default_factoryがMISSINGであることを検証した。assemblerの`version=1`は既存方式と同じprotocol版で、
runtimeの設定値ではない。

修正前に実際の`discover_contract_declarations`で対象symbolが未登録であることを記録した
([composition-before.json](../../runs/run-ball3d-topk-multihypothesis-20261103/composition-before.json))。
修正後は同じ既存APIの宣言集合への登録と`inspect_typed_adapter`による全16 fieldのrequired / composition-ownedを確認した
([composition-authority.json](../../runs/run-ball3d-topk-multihypothesis-20261103/composition-authority.json))。
さらに`test_topk_settings_are_discovered_as_required_composition_owned_fields`は実catalogの
`SOURCE_CONTRACT_DECLARATIONS.adapter_symbols`と`ADAPTER_CONTRACTS`を直接検査して成功した。
単なるcatalog parityではないため、既定値を再導入して型が探索から消える故障も検出する。
`src/utils/configuration`自体は変更していない。一般の探索漏れ（既定値のある型を宣言・nestedの両探索から外す点）は
このPRの範囲外として親へ報告する。

旧headのPython literal値と修正後のHydra値を全16項目・型まで比較して一致した。
`pipeline.yaml`も旧headとbytes一致、min_framesは5で不変。coreの全関数本体も旧headとAST一致した。
前半・後半のラベルを使った調整や方式・採点の変更は行っていない。

事前設計は [composition-test-plan.md](../../runs/run-ball3d-topk-multihypothesis-20261103/composition-test-plan.md)、
再現は [composition-tests.sh](../../runs/run-ball3d-topk-multihypothesis-20261103/composition-tests.sh)。
`--collect-only`は532件/77.63秒（wrapper 80.14秒）。`--durations=0 --durations-min=0`は
531 passed・既存の空parametrizeによる1 skip/298.83秒（wrapper 301.74秒）、すべて-n 0。
内訳は設定監査389 passed/1 skipと、元の関連suite・synthetic visualization・DINOの142 passed。
追加2nodeは監査登録0.002秒、必須topk scalar欠落拒否0.088秒。最大setupは0.113秒で共有compositionはmoduleに1回。
既存のcold-import2caseが83.09/83.47秒、全source監査が28.56秒で5秒目安を超えるが、
実processのimport順と全source/catalog契約の検証であり、対象削減やmockで代替しない。
全node/phaseは [composition-tests.json](../../runs/run-ball3d-topk-multihypothesis-20261103/composition-tests.json)、
件数と超過caseの要約はcomposition-summaryに保存した。変更Python11ファイルのruff・mypyは成功。

修正コードcommit df93a3e415e05f8c4e247ff537c579409ab5c106で固定bundleを基盤runnerから再実行し、
baseline/本方式×全view/3LOVOの計8結果でframe軸・位置・valid・used-viewと全診断配列がv2とbitwise一致した。
v2の129ファイルもサイズ・checksum一致。採点・camera・参照も不変なので誤差とcoverageは変わらない。
採点の再計算は行わず、上の負の結果と採否の判断を維持する。CPU 54.87秒、最大RSS635,220KiBで、
GPU・再推論・動画の再生成・GTの方式入力への利用はない。
新しい`outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/topk_multihypothesis_composition_verification_v1/`
には`verification.json`だけを保存した。基盤runnerの一時予測ファイルは照合後に削除し、既存成果物は変更していない。
同じ記録は [composition-prediction-check.json](../../runs/run-ball3d-topk-multihypothesis-20261103/composition-prediction-check.json)、
実パス・checksum・完全な再現コマンドはcomposition-summaryの`predictions`/`reproduction`に記録した。
最終HEADのCIはPR #1672に記録し、人間の採否までmergeしない。
knowledge validatorは`--check-summary --base-ref origin/main`で0 errors（既存219 warnings）、
repro path検査は0 missing（既存28 unverifiable）。run保存量は上限内で、予測配列はknowledgeへコピーしていない。

## 明示レビューcross-1: 共通点・追加点・喪失点の誤差

親の明示レビュー`parent-1554-5647a023173b48feac926b069d0a207c`（対象head 0055da23f）の指摘に対応した。
このPRで報告した条件は`topk_multihypothesis`の1件。別のTopK+LOESS条件は報告していない。
v2の`results/topk_multihypothesis/without_cam*`と、同じ比較で使った`results/pipeline_baseline/without_cam*`を
基盤`load_run`で読み、SHA-256をpinしたbundleと正解に対して基盤`error_samples(run, bundle, annotations)`を呼んだ。
誤差分布も基盤`distribution(threshold=20)`を使う。独自のprojection/scorerはなく、方式の再実行・設定調整・src変更はない。
集計コードcommitは842165d650857d8ad344f5abcd55a032fe8379fe。

主評価[505,1010)のLOVOを(held-out camera, frame)の組で突き合わせた。共通点の2列は同じ353点である。

| 集合 | 件数 | top-K 中央値/p90 px | top-K >20px率 | baseline 中央値/p90 px | baseline >20px率 |
|---|---:|---:|---:|---:|---:|
| 共通 | 353 | 9.274 / 34.073 | 17.85% | 8.778 / 27.825 | 15.01% |
| 追加（方式のみ） | 180 | 19.721 / 244.478 | 49.44% | — | — |
| 喪失（baselineのみ） | 472 | — | — | 10.255 / 26.681 | 17.37% |

追加点の89/180点が20px超で、方式全点の粗大誤差152点の58.55%を占める。
20px超率は共通点の63/353=17.85%から、追加点を含む全点の152/533=28.52%へ上がる。
全点p90は103.60pxで、共通点34.07pxより大きい。中央値・p90は加重平均できないが、
追加点側の長い誤差分布を分けて見ると、coverageの増分の質を確認できる。
共通点でもbaselineのp90 27.83pxより悪化している。
喪失472点のうち390点はbaselineで20px以内であり、保持できた点と追加点だけから性能を判断しない。
採点数-292点は追加180点と喪失472点の差である。この集合分解は記述統計で、運動コスト等の原因分離ablationではない。
1clip・近似校正と人間2D参照という制限、悪化の結論、既定・採否の判断を維持する。

| 区間 | 共通 | 追加 | 喪失 | 方式全点 = 共通+追加 | baseline全点 = 共通+喪失 |
|---|---:|---:|---:|---:|---:|
| 主[505,1010) | 353 | 180 | 472 | 533 | 825 |
| 前半[0,505) | 372 | 217 | 449 | 589 | 821 |
| 全[0,1010) | 725 | 397 | 921 | 1122 | 1646 |

全3区間×(pool+3camera)×(全参照+observed+estimated)の36層、72件の件数検算はすべて成立した。
既存`paired-scores.json`の全点・2方式共通点について、count/median/p90/>20px率の48値が完全一致した。
空集合の中央値・p90・率はnullで、0pxの結果として扱わない。読み出した比較元68ファイルのchecksumは前後で不変。
bundle/正解は基盤loaderのexpected_sha256で照合し、saved runも元manifestの外部pinで照合した。

出力先は新設の
`outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/topk_multihypothesis_cohort_v1/`。
`cohorts.json`、全層の`cohorts.csv`/`cohorts.md`、camera/frame別の集合・誤差を記した`samples.csv`、
checksum目録`artifacts.json`を保存した。実パス・全checksumは
[cohort-artifacts.json](../../runs/run-ball3d-topk-multihypothesis-20261103/cohort-artifacts.json)、
小さな全層指標は[cohort-metrics.json](../../runs/run-ball3d-topk-multihypothesis-20261103/cohort-metrics.json)と
[cohort-tables.csv](../../runs/run-ball3d-topk-multihypothesis-20261103/cohort-tables.csv)。
point別CSVはoutputsに置き、予測配列はknowledgeへコピーしていない。CPU集計1.52秒、最大RSS633,668KiB。

再現は同checkoutから新しい`--output`を指定する。入力の実パスとpinはcohort-metricsのprovenanceにも保存した。

```bash
TOPK_FIXED=/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  knowledge/runs/run-ball3d-topk-multihypothesis-20261103/cohort_report.py \
  --bundle "$TOPK_FIXED/fixed_input_bundle_v1" \
  --annotations "$TOPK_FIXED/human_annotations/ground_truth_v1/ground_truth.json" \
  --comparison "$TOPK_FIXED/topk_multihypothesis_v2" \
  --output "$TOPK_FIXED/topk_multihypothesis_cohort_reproduction"
```

テスト設計は[cohort-test-plan.md](../../runs/run-ball3d-topk-multihypothesis-20261103/cohort-test-plan.md)。
新規5caseは行順、同frame別camera、504/505/1009、参照kind、20px境界、空集合、重複、件数不整合を検証する。
基盤scoringの既存13caseを含む18caseはcollect-only 4.84秒、18 passed/10.14秒（すべて-n 0）。
全node/phaseはcohort-test-results、要約はcohort-test-summary。最大case0.088秒、最大setup0.0092秒で基準内。
共通点の方式誤差をbaseline誤差へ取り違える故障注入は、期待中央値14.5に対する8.5の不一致で失敗し、
元bytesへ復元すると同nodeが成功した（cohort-mutation）。ruff/mypyは変更Python2ファイルで成功。
共有memoryへの追加は親の指示で取り消し、指定3ファイルをreviewed head 0055da23fと同じ状態へ戻した。
cross-1はreviewer agentの指摘を親が差し戻したもので、ユーザーの判断として記録した出典は誤りだった。
`cohort-memory-doc-*.json`とcohort-test-summaryのmemory関連欄は、取り消した変更に対する過去の実測記録として残す。
各記録の`shared_memory_change_status`に取消済み・過去記録であることを明示した。
取消理由と復元確認は[cohort-memory-retraction.json](../../runs/run-ball3d-topk-multihypothesis-20261103/cohort-memory-retraction.json)。
cohortの集計コード・テスト・数値・成果物は変更していない。
knowledge validatorは`--check-summary --base-ref origin/main`で0 errors（既存219 warnings）、
repro path検査は0 missing（既存28 unverifiable）。
最終headのCI結果はPR #1672へ記録し、統合比較の採否までmergeしない。
