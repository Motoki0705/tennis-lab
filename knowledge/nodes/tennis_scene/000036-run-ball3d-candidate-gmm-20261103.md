---
id: run-ball3d-candidate-gmm-20261103
type: run
task: tennis_scene
sequence: 36
recorded_at: '2026-11-03'
title: 候補GMMはclip_000でLOVO悪化・全view欠測増加・不確かさ過小を確認
issue: 1563
provider: codex
date: '2026-11-03'
status: done
parents:
- run-ball3d-foundation-review-20261103
config:
  top_k: 3
  primary_frames:
  - 505
  - 1010
  tuning_frames:
  - 0
  - 505
  prior_mean_m:
  - 0.0
  - 0.0
  - 2.0
  prior_std_m:
  - 8.0
  - 16.0
  - 6.0
  gpu_used: false
  detector_inference_rerun: false
repro:
  branch: codex/ball3d-candidate-gmm
  commit: 5f699863a26a30c9cbcb518569954434f3694542
  command: bash knowledge/runs/run-ball3d-candidate-gmm-20261103/repro.sh BUNDLE_DIR
    VIDEOS_DIR GROUND_TRUTH_JSON FOUNDATION_V2_DIR NEW_OUTPUT_DIR
artifacts:
  run_dir: knowledge/runs/run-ball3d-candidate-gmm-20261103
  output_dir: outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/candidate_gmm_v2
  metrics: knowledge/runs/run-ball3d-candidate-gmm-20261103/metrics.json
  folds: knowledge/runs/run-ball3d-candidate-gmm-20261103/fold-metrics.csv
  uncertainty: knowledge/runs/run-ball3d-candidate-gmm-20261103/uncertainty-coverage.csv
  tests: knowledge/runs/run-ball3d-candidate-gmm-20261103/test-summary.json
  inventory: knowledge/runs/run-ball3d-candidate-gmm-20261103/inventory-verification.json
  cohort_analysis: knowledge/runs/run-ball3d-candidate-gmm-20261103/cohort-receipt.json
metrics:
  annotation_oracle:
    primary_lovo_median_px: 7.8586408238936745
    primary_lovo_p90_px: 20.721912427346656
    primary_lovo_count: 1345
    primary_lovo_coverage: 0.9199726402188783
  candidate_gmm_map_none:
    primary_lovo_median_px: 16.201927101619646
    primary_lovo_p90_px: 255.71869056756225
    primary_lovo_count: 927
    primary_lovo_coverage: 0.63406292749658
  candidate_gmm_map_rts:
    primary_lovo_median_px: 19.001268901205066
    primary_lovo_p90_px: 149.86353371499987
    primary_lovo_count: 930
    primary_lovo_coverage: 0.6361149110807114
  candidate_gmm_mean_none:
    primary_lovo_median_px: 24.99931002263351
    primary_lovo_p90_px: 213.2603335791368
    primary_lovo_count: 928
    primary_lovo_coverage: 0.6347469220246238
  candidate_gmm_mean_rts:
    primary_lovo_median_px: 16.780366363186875
    primary_lovo_p90_px: 95.31116301913217
    primary_lovo_count: 930
    primary_lovo_coverage: 0.6361149110807114
  pipeline_baseline:
    primary_lovo_median_px: 9.340188442451916
    primary_lovo_p90_px: 27.078044438775773
    primary_lovo_count: 825
    primary_lovo_coverage: 0.5642954856361149
  tests_passed: 121
---

## 2026-11-05の人間の採否決定とコードの所在

[親 #1554の採否決定・最新合意revision 7](https://github.com/Motoki0705/tennis-lab/issues/1554#issuecomment-6652565114)でbaseline維持が決定した。
誤差分解の診断（PR #1670）だけをmainへmergeし、6方式のPRと統合比較PR #1682はmergeせずcloseした。
このノードに対応する[PR #1674](https://github.com/Motoki0705/tennis-lab/pull/1674)も未mergeでclose済み。branchは再現のため残している。
人間は「heatmapの精度そのものが悪く、baselineを超えるには検出モデルの学習戦略の改善が必要」と判断し、
後処理・三角測量の側で改善策を試す後継作業（#1683）をキャンセルした。以下の実測値と研究上の結論は変更していない。

`repro.commit` は当時の再現コード版 `5f699863a26a30c9cbcb518569954434f3694542` を指す。
元branchは `codex/ball3d-candidate-gmm`（close時head `a1266d23b3aaaa95fab3fbd1a25021295507d937`）。
このノードとrun bundleの取り出し元は統合branch `codex/ball3d-integrated-comparison` の
[commit `784dce5a15038965818f82933f9d13c92d50e143`](https://github.com/Motoki0705/tennis-lab/tree/784dce5a15038965818f82933f9d13c92d50e143)。
mainにない方式固有のsrc・testsは、元branchの履歴とこの統合commitに保存されている。
run内のreproduce/repro・再採点・検証scriptと本文のコマンドは、そのコードを必要とする歴史的な再現記録で、main上でそのまま実行できるものではない。
再現・再採点時は保存元の統合commitをcheckoutした環境で、本文の固定入力・実行版・byte照合条件を確認する。run bundle自体は取り出し元と同じbytesを保持した。
## 比較条件と実装

親 #1554 revision 6、子 #1563。公開meiji_3cam/video_000/clip_000の
3camera・1010frameのみを固定bundleからCPU replayする。基盤のloader、
全view/LOVO runner、型付き保存/読込、採点と共通点比較、動画/図/HTMLを使う。
正解は#1668 readerで検証した直接観測2589点と承認推定370点。
cam0末尾外挿28点・座標なし43点は補完せず、元kindとイベント25行を保持する。

前半[0,505)だけで共分散倍率とscore関数の傾き・切片をfitする。
各LOVOは保持cameraだけの前半正解を使う。後半の正解はfitに使わない。
局在候補の共分散倍率は、20px以内の最寄り候補でMahalanobis距離のGaussian scale MLE、
score関数は採点可能な前半点で「top-3内に20px以内の候補がある」をtargetとする。
これは利用可能な候補支持の代理値で、物理的存在確率の独立較正ではない。
較正値を予測前に固定し、後半の主指標による再調整は行わない。
正解やその欠測mask、除外viewをproductionの入力に渡さない。

方式・単位・設定・maskの正本は [候補GMM README](https://github.com/Motoki0705/tennis-lab/blob/784dce5a15038965818f82933f9d13c92d50e143/src/tennis_scene/ball_gmm/README.ja.md)。
joint log曲率の交差項を含む3×3 fit、score順の最初の有効3候補を既存方式Aへ渡す。
等方fallbackはなく、明示した固有値上下限と棄却数を記録する。
コートmのGaussian priorを固定し、2camera以上のactive支持に条件付けたGMMから
最大重み成分MAPと混合momentを出す。MAP共分散は局所成分内、meanは成分間分散を含む。
0/1cameraだけの質量を三角測量観測へ数えない。全成分を処理できないframeは理由付き欠測。
等速度Kalman/RTSは欠測でresetし、maskを変えずgapを埋めない。

## 先行結果との関係

[#1670の誤差内訳](https://github.com/Motoki0705/tennis-lab/pull/1670)には、
argmax失敗の後半57.2%でtop-8に回収余地がある一方、既存maskを保つ候補oracleの
3D改善は小さいという結果がある。回収余地はGMMの正しいmode選択の保証ではない。
[#1672のtop-K時間探索](https://github.com/Motoki0705/tennis-lab/pull/1672)の本文と
knowledgeを読み、null状態/時間beamで欠測を増やさない設計にした。
今回も幾何・最適化の失敗で欠測が増える可能性は別に残るため、
誤差と採点数・coverageを同じ重さで報告する。#1559の2D decodeには依存しない。

## 解釈の限界

1clip・近似pinhole校正・独立3D正解なしの比較である。
投影共分散は一次Jacobianによるposteriorだけで、正解の推定ノイズ、校正誤差、
非線形tailやMAPのmode選択誤りを含まない。楕円被覆率は可用性を条件とする。
Kalman/RTSの等速度仮定は打球やバウンドを丸めうる。
scoreの支持確率代理を物理的存在の確率と同一視しない。
学習runではないためTensorBoardやseed間学習曲線は適用しない。
統合比較後、2026-11-05に人間がbaseline維持と本方式の不採用を決定し、PR #1674はmergeせずcloseした。

## 主結果と被覆率

**今回の4方式はいずれもbaselineよりLOVO誤差が悪化した。採用は勧めない。**
採点可用率はMAPで56.43%→63.41%へ増えた一方、全view有効3Dは420/505→62/505へ減った。
主評価のMAP差は中央値+6.862px、p90+228.641px、>20px率+28.620pp、採点+102点・coverage+6.977pp。
後半を見た再調整はない。各fold・区間・全viewの表は [fold-metrics.csv](../../runs/run-ball3d-candidate-gmm-20261103/fold-metrics.csv)。

| 区間 | 方式 | LOVO median/p90 px | >20px率 | 採点数/対象数・coverage |
|---|---|---:|---:|---:|
| 主 [505,1010) | pipeline_baseline | 9.340/27.078 | 16.36% | 825/1462・56.43% |
| 主 [505,1010) | candidate_gmm_map_none | 16.202/255.719 | 44.98% | 927/1462・63.41% |
| 主 [505,1010) | candidate_gmm_mean_none | 24.999/213.260 | 57.33% | 928/1462・63.47% |
| 主 [505,1010) | candidate_gmm_map_rts | 19.001/149.864 | 48.92% | 930/1462・63.61% |
| 主 [505,1010) | candidate_gmm_mean_rts | 16.780/95.311 | 45.91% | 930/1462・63.61% |
| 主 [505,1010) | annotation_oracle | 7.859/20.722 | 11.15% | 1345/1462・92.00% |
| 前半 [0,505) | pipeline_baseline | 10.716/28.483 | 16.44% | 821/1497・54.84% |
| 前半 [0,505) | candidate_gmm_map_none | 15.428/239.741 | 41.65% | 1203/1497・80.36% |
| 前半 [0,505) | candidate_gmm_mean_none | 24.331/194.599 | 54.86% | 1203/1497・80.36% |
| 前半 [0,505) | candidate_gmm_map_rts | 21.661/213.446 | 52.29% | 1203/1497・80.36% |
| 前半 [0,505) | candidate_gmm_mean_rts | 18.483/129.417 | 47.05% | 1203/1497・80.36% |
| 前半 [0,505) | annotation_oracle | 9.655/22.524 | 11.46% | 1422/1497・94.99% |
| 参考 [0,1010) | pipeline_baseline | 10.208/28.196 | 16.40% | 1646/2959・55.63% |
| 参考 [0,1010) | candidate_gmm_map_none | 15.622/245.344 | 43.10% | 2130/2959・71.98% |
| 参考 [0,1010) | candidate_gmm_mean_none | 24.661/200.567 | 55.94% | 2131/2959・72.02% |
| 参考 [0,1010) | candidate_gmm_map_rts | 20.825/195.365 | 50.82% | 2133/2959・72.09% |
| 参考 [0,1010) | candidate_gmm_mean_rts | 18.026/102.217 | 46.55% | 2133/2959・72.09% |
| 参考 [0,1010) | annotation_oracle | 8.790/21.392 | 11.31% | 2767/2959・93.51% |

全6方式の主評価共通LOVOは514/1462点（35.16%）。baseline 9.282/30.689pxに対し、
MAP 11.172/47.968px、mean 17.082/84.395px、MAP-RTS 11.670/60.319px、
mean-RTS 10.603/43.519pxで、共通集合でも悪化。2方式だけの共通525点でも
baseline 9.356/34.980px、MAP 11.477/51.991px。可用性の条件付きで、欠測点の誤差は回復しない。

| 方式 | 主評価n | 50%楕円被覆 | 90%楕円被覆 | 95%楕円被覆 |
|---|---:|---:|---:|---:|
| candidate_gmm_map_none | 927 | 19.96% | 52.00% | 57.39% |
| candidate_gmm_mean_none | 928 | 52.80% | 78.12% | 81.57% |
| candidate_gmm_map_rts | 930 | 4.19% | 14.09% | 17.10% |
| candidate_gmm_mean_rts | 930 | 9.25% | 29.89% | 38.82% |

[区間/fold/参照kind別の被覆率表](../../runs/run-ball3d-candidate-gmm-20261103/uncertainty-coverage.csv) と
[不確かさ図](../../runs/run-ball3d-candidate-gmm-20261103/uncertainty.png) を保存した。
RTSではposteriorが縮んでも誤差を覆わず、滑らかさを精度・較正の改善とは扱わない。

## 欠測の原因と当時の検証候補（追加改善はキャンセル済み）

ラベルを使わない診断では、全viewの688/1010frame（主評価443/505）が、方式Aの
「active cameraの背面にmodeがある」エラーでframe全体の欠測となった。
cam0/1/2除外では同じ理由が549/310/0frame。全frame棄却は実装の明示方針で、
失敗成分だけを黙って落としていない。時間beam/null探索による欠測ではない。
4版のvalid maskは同一で、LOVO採点数の微差は投影の前面可用性の差である。
候補間の曖昧さ・conditional融合・近似校正・局所Gaussianの寄与を独立ablationしていないため、
個々へ悪化量を割り当てない。未使用clipまたは前半だけでK/支持モデル/失敗扱いを
独立に検討する案は当時の未検証候補として残すが、2026-11-05に後処理側の追加改善はキャンセルされた。この後半で設定を選び直していない。

## 再現・検証

実装前の [test-plan.md](../../runs/run-ball3d-candidate-gmm-20261103/test-plan.md) に契約・故障・層・既存node・コストを固定した。
新規26caseと既存95case。非対称camera/frame、既知の等方/回転Gaussian、手計算Kalman更新、
20pxの局在支持・前半較正の隔離、pinhole Jacobian、欠測reset、main固定配列、実codec/LOVOを検証する。
既存media integrationに共分散と追加図を加え、別の重い動画fixtureを増やしていない。
全121nodeとphase実測は [test-results.json](../../runs/run-ball3d-candidate-gmm-20261103/test-results.json)、
集約は [test-summary.json](../../runs/run-ball3d-candidate-gmm-20261103/test-summary.json)。削除・統合はない。

数値再生のcommitはc4112da3f、v2媒体のcommitは5f699863a。v1の全20予測recordを
基盤の型付きloaderで再読込し、v2の全field/全診断配列・dtype、runtimeを除く全採点値、
較正値・不確かさの完全一致を確認した。予測やパラメータの再調整ではなく表示の改善である。
baselineの4foldはfoundation_v2と位置/valid/used-view配列がbitwise一致。
元store照合もvalid/used-view mask一致・3D最大差2µm許容内。

v1のscript timerは1134.70秒、GNU time wallは1078.99秒、最大RSS1,051,428KiB。
v2の再表示はscript timer180.15秒・GNU time wall175.94秒、最大RSS1,188,496KiB。
異なる時計の原値を資源receiptに保存した。共有hostでの観測であり速度benchmarkとはしない。
GPU・再推論・学習はない。352ファイルの全サイズ/checksum・未宣言ファイルなしを確認した。
比較MP4は1200×1150・1010frameを全decodeし、HTML参照切れ0。
代表画像、frame504のcamera/top/side楕円、availabilityだけで選んだ127/387/614/886の3D楕円を目視確認した。
cam1 unknownとcam0末尾外挿除外が正解markerとして描かれないことを確認した。
配列・動画・HTMLはoutputs、小さな図/指標/再現証拠はknowledgeに残す。

全文再生はreproの5引数で新規directoryを指定する。表示だけの再実行は
`run.py` の同じ引数に `--reuse-predictions OLD_OUTPUT_DIR` を加える。
較正値・method config・bundle identityの一致を検証し、保存した結果の元code provenanceを保持する。

最終 `--collect-only` は121件/6.87秒、`--durations=0 --durations-min=0` は121 passed/28.76秒（-n 0）。最大case2.714秒・setup0.01184秒で5秒/1秒の基準内。交差曲率を消す故障注入で専用nodeの数値assertが失敗し、元bytes復元後に成功した。ruff/mypy成功、knowledgeは0 errors（既存219 warnings）、repro pathは0 missing（既存28 unverifiable）。

## PR #1674 のCI修正

対象の旧HEADは2fd5385ab、CI run 41611770627。設定dataclassのPython既定値で
設定catalogの収集が止まったため、CandidateGMMConfigの全16fieldと評価adapterの
暗黙settings instanceを削除した。Hydra YAMLは変更せず、実行入口とunit fixtureも
compositionから明示的に設定を渡す。監査の除外・緩和は行っていない。

既定の再投影診断は、float32で保存する位置に丸める前のfloat64 fit.xから計算される。
BLAS core typeをSandybridgeへ切り替える逐次probeで、float64解が最大4.0701e-13m変わり、
位置・valid・inliers・採用UV・理由コードは完全一致したまま、残差7要素が1 ULP変わることを再現した。
最大差は1.4552e-11px。CIの7要素/最大3.6380e-12pxの差と同じ種類の現象で、
このPRの既定演算の変更ではない。既定経路6sourceはorigin/mainとbytes一致する。

回帰テストは上の5種類の配列の完全一致を維持し、再投影診断だけ4 float32 ULP以内
（このfixtureでは最大5.8208e-11px未満）を許容し、ゼロ残差は完全一致のままにした。
LOVO等の採点は保存位置から再投影して行うため、この診断の許容幅は方式比較を変えない。
根拠は [ci-fix-numerical-evidence.json](../../runs/run-ball3d-candidate-gmm-20261103/ci-fix-numerical-evidence.json)。

実データの3D予測全体は再実行していない。旧Python既定値と新compositionの16値が等しく、
設定validatorとadapter実行部のAST、全数値kernelのbytesが不変であることを確認した。
さらに固定bundleの3camera×1010frameからGaussian化を再実行し、21診断配列の全要素・dtypeが
保存済みv2と完全一致した。したがって設定の渡し方だけの変更であり、3D数値・採点結果は不変と判断する。
`candidate_gmm_v2/` は変更せず、目録checksumも維持する。根拠は
[ci-fix-invariance.json](../../runs/run-ball3d-candidate-gmm-20261103/ci-fix-invariance.json)。

設定監査・GMM関連・結合・DINO extensionは435 passed/396.17秒（-n 0）。
時間の大半は既存のfresh-process import 2件（115.68/104.60秒）とrepo全体監査（42.45秒）で、
その契約上cold import/全source走査を省けない。新しい重いfixtureは追加していない。
基盤等の関連テストと以前収集で止まったsynthetic可視化契約は126 passed・既存の空parameter 1 skip/102.39秒。
skipを成功件数には数えない。全phaseの実測は
[ci-fix-test-results.json](../../runs/run-ball3d-candidate-gmm-20261103/ci-fix-test-results.json)。
この時点では指定DINO分離テストは失敗しておらず、a8982a7e0の変更は取り込んでいない。
採用判断・LOVO悪化/被覆不足という研究結論は変わらない。

## cross-1: 共通・追加・喪失点の誤差（2026-11-04追記）

明示レビューrun `parent-1554-5647a023173b48feac926b069d0a207c` の指摘に対応した。
方式の再実行・設定の再調整・srcや基盤scoringの変更はない。
解析コードcommitは `1b171a1cd11d506e66d7a54a23e434ab844ac176`。
v2の目録SHAからpinした4条件と同じbaselineの保存済み15 LOVO recordを
基盤の `load_run` で読み、SHAを照合したbundle/GTと
`error_samples(run, bundle, annotations)` で採点点を得た。
誤差の分布も基盤の `distribution` を使い、独自の再投影や採点は作っていない。

各条件対baselineの `(held-out camera, source frame)` を突き合わせた。
全6方式共通の514点とは異なる2方式間の共通集合である。
以下は主評価[505,1010)、直接観測・承認推定をpoolした値。単位はpx、>20は厳密な超過。

| 条件 | 共通n | 共通・方式 median/p90/>20% | 共通・baseline median/p90/>20% | 追加 n; median/p90/>20% | 喪失 n; baseline median/p90/>20% |
|---|---:|---:|---:|---:|---:|
| MAP | 525 | 11.477/51.991/28.95% | 9.356/34.980/21.71% | 402; 38.460/363.652/65.92% | 300; 9.116/17.005/7.00% |
| mean | 525 | 17.677/90.297/46.29% | 9.356/34.980/21.71% | 403; 46.996/333.982/71.71% | 300; 9.116/17.005/7.00% |
| MAP-RTS | 526 | 11.897/66.985/32.13% | 9.348/34.952/21.67% | 404; 48.102/228.605/70.79% | 299; 9.163/17.045/7.02% |
| mean-RTS | 526 | 10.754/46.186/30.04% | 9.348/34.952/21.67% | 404; 35.128/133.510/66.58% | 299; 9.163/17.045/7.02% |

MAPの採点純増102点は402点追加と300点喪失の差で、良い予測102点の回収を意味しない。
追加点の265/402点が20px超である一方、失ったbaseline点は279/300点が20px以内だった。
MAPの共通点の誤差11.477/51.991pxに対し、追加点は38.460/363.652pxと大きく、
全採点点の16.202/255.719pxを悪化側へ動かしている。
baselineの全点9.340/27.078pxと共通点9.356/34.980pxの差にも、誤差の小さい点の喪失が含まれる。
20px超数はbaseline全体135点からMAP全体417点へ増えた。
差282点は共通での38点増加（152−114）と追加265点から喪失21点を引いた数に一致する。
分位点は加算分解できないため、中央値/p90の悪化をこの件数式へ換算してはいない。
4条件とも共通点でも悪化し、追加点の中央値35.128–48.102px・p90 133.510–363.652pxも大きい。
coverageの増加を精度改善とする根拠はなく、採用を勧めない結論は維持する。

MAPの追加はcam0/1/2で43/179/180点、喪失は233/66/1点と非対称である。
追加の参照kindは直接観測393点・承認推定9点、喪失は230/70点。
全区間・前半・後半、camera別、kind別の全分布と、各点のidentity/誤差を保存した。
欠測点の未知の方式誤差は補わず、これも可用性に条件付きの比較である。

保存先は
`/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/candidate_gmm_cohort_v1/`。
`cohorts.json` に144条件行、`cohorts.csv` に共通双方/追加/喪失と全点の分布、
`cohorts.md` に区間別pool表、`samples.csv` にcamera/frame/kind・集合名・双方の誤差を残した。
目録 `artifacts.json` のSHA-256は
`ea0c60de9c9366464786a15142f20bbfb317b224a69af7345b2498bf263a1fbc`。
各ファイルと15入力recordのchecksumは
[cohort-artifacts.json](../../runs/run-ball3d-candidate-gmm-20261103/cohort-artifacts.json)、
検算・資源・テストのreceiptは
[cohort-receipt.json](../../runs/run-ball3d-candidate-gmm-20261103/cohort-receipt.json)。
4ファイルのサイズ/SHA・未宣言ファイルなし、v2目録のSHA不変を再照合した。

全144行で共通+追加=方式全点、共通+喪失=baseline全点の計288式が一致した。
主評価の方式全点はMAP/mean/MAP-RTS/mean-RTSで927/928/930/930点、baselineは全て825点。
既存v2の全点件数/中央値/p90/最大/超過率の1440値が完全一致し、camera/kind/区間の加算792件も一致した。
不一致・重複・参照kind不整合でscriptは例外停止する。解析11.84秒・最大RSS638,976KiB、GPU・動画処理なし。

実装前の [cohort-plan.md](../../runs/run-ball3d-candidate-gmm-20261103/cohort-plan.md) に沿って
非対称camera/frame、境界505、厳密な>20px、空集合、重複・kind不整合を検証した。
新規3件と既存scoring 13件の `--collect-only` は16件/4.42秒、
`--durations=0 --durations-min=0` は16 passed/4.34秒（-n 0）。
新規caseの最大setupは0.01088秒、setup+call+teardown最大0.01382秒。
全node/phaseは [cohort-tests.json](../../runs/run-ball3d-candidate-gmm-20261103/cohort-tests.json)。
cameraをjoinキーから落とす故障注入は中央値15.5≠21pxのassertで失敗し、bytes復元後は全16件成功した。

以下は保存元branch／統合commitのcheckout rootで使う再現コマンドで、mainでは方式実装が不足する。
新規出力先を指定する（bundle/GT/v2のSHAはscriptに固定）。

```bash
BALL3D_INPUT=/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python knowledge/runs/run-ball3d-candidate-gmm-20261103/cohorts.py \
  --bundle "$BALL3D_INPUT/fixed_input_bundle_v1" \
  --annotations "$BALL3D_INPUT/human_annotations/ground_truth_v1/ground_truth.json" \
  --comparison "$BALL3D_INPUT/candidate_gmm_v2" --output "$BALL3D_INPUT/candidate_gmm_cohort_replay_v2"
```
