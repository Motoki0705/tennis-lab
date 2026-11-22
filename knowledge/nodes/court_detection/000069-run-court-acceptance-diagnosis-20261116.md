---
id: run-court-acceptance-diagnosis-20261116
type: run
task: court_detection
sequence: 69
recorded_at: '2026-11-16'
title: Meiji 4camera全276件の採否診断と後処理改善planning
issue: 2190
provider: codex
session: 01a1e30b-a305-7672-8355-525787fd6380
date: '2026-11-16'
status: done
summary_impact: none
config:
  phase: planning
  checkpoint: court_detection/multiscale_depth3/b863df1f01f0.ckpt
  dataset: tennis_multiview/processed/meiji_4cam/dataset
  source_frame_index: 0
  resource: half
  device: cuda
  reference_approved: false
  inference_changed: false
metrics:
  reviewed_images: 276
  old_accepted: 152
  old_rejected: 124
  rejection_inliers_only: 79
  rejection_outside_only: 15
  rejection_both: 30
  recaptured_selected_regions: 10
  recaptured_full_frame_regions: 1
  matching_old_selected_status_and_count: 10
  existing_tests_passed: 58
  existing_tests_wall_seconds: 13.26
repro:
  commit: 2736a24a3fa643254f441f318ed9241bc2c73724
  branch: codex/court-acceptance-postprocess
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python
    -u outputs/tennis_scene/evaluate/court-acceptance-planning-20261116/capture_stages.py
artifacts:
  run_dir: knowledge/runs/run-court-acceptance-diagnosis-20261116
  log: .training_queue/logs/0-8205189007555071410_2026-11-16T06-36-32-444928589Z_2702112_court-acceptance-planning-stages-a2.log
  output_dir: .worktree-outputs/court-acceptance-postprocess/tennis_scene/evaluate/court-acceptance-planning-20261116
parents: [run-court-meiji-model-only-regions-20260922]
relations:
- to: run-court-acceptance-stage-probe-missing-assets-20261116
  rel: related
  note: 初回は資産参照で失敗。共有資産を明示して再実行した。
- to: run-court-hybrid-fit-search-20261021
  rel: compares
  note: 同じ既定checkpointの既存val診断。独立testの証拠ではない。
papers: []
tags: [planning, region-search, partial-court, hybrid-homography]
---

これは#2190のplanning成果であり、採否・後処理の実装や閾値変更は行っていない。
[親revision 3の合意](https://github.com/Motoki0705/tennis-lab/issues/2150#issuecomment-6909050876)と
[#2190](https://github.com/Motoki0705/tennis-lab/issues/2190)を確認した。
既存branchを使い、PR・追加worker・validatorは起動していない。参照草案は未確認のため、
正誤の混同行列、参照に対するpx誤差、改善率はまだ計算していない。

## 1. 診断と分類

元証拠はmain checkoutの`outputs/tennis_scene/evaluate/meiji4cam-court-review-20261116/`。
全276件・3312領域候補を集計し、全24枚のcamera別contact sheetを目視した。
下表の不採用理由は、元scriptが表示した最良候補についての分類である。

| camera | 採用 | 不採用 | inlier不足のみ | 画像外のみ | 両方 |
|---|---:|---:|---:|---:|---:|
| cam0 | 29 | 40 | 40 | 0 | 0 |
| cam1 | 63 | 6 | 6 | 0 | 0 |
| cam2 | 52 | 17 | 17 | 0 | 0 |
| cam3 | 8 | 61 | 16 | 15 | 30 |
| 合計 | 152 | 124 | 79 | 15 | 30 |

不採用候補のinlierは7点69件、6点29件、5点9件、4点2件。残り15件は8点以上で画像外のみ。
表示対象の276候補はすべてhybridの`status=ok`、凸、面積条件を通っている。
一方、全3312候補では`ok`1692、候補なし704、joint optimization失敗563、
LINE支持不足348、曖昧5であり、表示されなかった候補の失敗と最終gateを混同しない。

人間の参照確認前の**目視による仮分類**は次のとおり。許容pxが未決であり、
「概ね整合」を正解ラベルや採用可能の証明として扱わない。

| camera | 採用：概ね整合 / 明瞭なずれ / 保留 | 不採用：概ね整合 / 明瞭なずれ / 保留 |
|---|---:|---:|
| cam0 | 29 / 0 / 0 | 38 / 2 / 0 |
| cam1 | 63 / 0 / 0 | 6 / 0 / 0 |
| cam2 | 52 / 0 / 0 | 17 / 0 / 0 |
| cam3 | 0 / 0 / 8 | 0 / 11 / 50 |

不採用の61件は見た目に概ね整合する。cam0 clip_013・017はfar側が潰れており、
inlier不足を一律に解除すると救済対象に誤推定を含める。
cam3の明瞭なずれはclip_001・004・006・009・010・013・017・020・023・027・058。
採用8件を含む残り58件は、部分観測・細い線・小さなずれの許容が未確定で保留した。
したがって「誤った推定を通す件数」はこの段階で確定しない。
全件の行別記録はrun bundleの`visual_review.csv`、代表8例はoutputsの
`representative_examples.jpg`に保存した。

原因は3段階に分かれる。

- **最終採否**：`inference/regions.py::select_court_region`が、14点の
  `visible.all()`と`inside_content`を要求する。表示の理由は「corners」だが実装は14点すべて。
  正しい画像外投影を採用できない。既存の下流`calibrate_local_courts`は画像外のH投影点を
  `require_in_image=False`で既に扱えるため、このgateと下流の契約が食い違う。
- **支持点数と順位**：領域gateは元画像対角の0.005（1920×1080で約11.0px）以内の
  raw点数を数え、点数→confidence合計で候補を選ぶ。hybridのfit点は最大8個で、
  この再集計のinlier数とは別物。fitに不使用の点も数えられ、支持点の独立性・分布・
  外挿安定性を判定しない。hybrid内部の距離閾値はcrop対角基準なので、領域間で実pxも異なる。
- **候補生成・joint fit**：単一peakとscoreで有効になったraw KPが、元画像端やcrop端付近に
  集まる。PROSAC・4点候補が歪む一方、既存のtrimmingはその一部を最終fitから除いている。
  LINEはcrop内で切り取られ、画像外を負例にせず、平均支持と縦横2本ずつ等で判定する。
  これは必要な設計だが、見えていないbaselineの位置を十分に拘束できない候補も通る。

現行設定を変えず、選択済み10crop＋cam3 clip_000のfull-frameをGPUで記録した。
選択済み10件のstatus・inlier数は元summaryと一致した。
outputsの`stages/`のJSONにはKP-only、LINE選択、seed、最終H、fit集合履歴を保存した。
Gitのrun bundleには、点集合・距離・LINE支持等を抽出した`stage_metrics.json`を残す。

| 代表例 | 元の採否 | 現行fitで観測したこと |
|---|---|---|
| cam0 clip_000 | 不採用、7点 | 最終fitは6点。LINE forward 0.839 / reverse 0.994、9線支持。点数だけで拒否された。 |
| cam0 clip_013 | 不採用、4点 | 最終fitは3/5/7/11。far側が歪んでも平均LINE forward 0.744 / reverse 0.897でhybrid成功。 |
| cam3 clip_000 | 採用、8点 | raw KP2/3は約2.1pxに重なる。最終fitは6/8/9/10/13で、2/3/5/7は不使用。LINE選択から最終fitでもずれが残る。 |
| cam3 clip_013 | 不採用、6点＋画像外 | cropは[0,0,1440,540]。KP3/7/11が右端付近、2/5が下端付近。最終fitは9/10/12/13だけ。近baselineのLINE支持は0、8線支持でhybrid成功。最終の近四隅は約(2010,1095)/(3719,431)へ外挿される。 |

代表cam3では**端の点を最終目的関数へ入れ続けたことが直接原因とは言えない**。
rawの誤対応に加え、観測範囲・線の意味対応・外挿の弱い拘束が原因候補。
反実仮想による要因分離は未実施であり、端点除外だけの改善効果は主張しない。
別形状の候補が十分残らず`alternative_score_gap=null`でも成功する例があり、
これを「解が一意」と読み替えるべきではない。

## 2. 人間に確認してもらう参照

モデル出力をGTにせず、原画像の可視地面線の交点を手で読む。各cameraで8〜9点を
仮指定し、Hを最小二乗で作った**未確認の草案**をoutputsの`reference_draft/`へ置いた。
`references.json`は`human_reviewed=false`、`approved_for_evaluation=false`。
精密な注釈ではなく、特に遠方の短い線とcam3の右端は修正確認が必要である。

人間には4枚の`cam0_review.jpg`〜`cam3_review.jpg`を見せる。
各1枚はclip_000・034・068の3時点について、元画像／同じ参照Hの重ね描きを並べたもの
（計12元画像・12overlay）。青は地面の9本の線、黄十字とIDは手指定点。
net上端を地面の参照に使わない。各overlayの1920×1080版も保存した。
cam3には右端・近baselineの拡大と不確かな点を追加確認し、画面外の点を端へ置かない。

確認時は、fit用の交点に加えて未使用の線上位置を確認し、同一Hの再投影だけで
正確性を自己証明しない。開始・中盤・終盤の静止性を確認し、動きがあればcamera単位の
1参照を無条件に適用しない。承認者・日付・参照版・コメントURLを保存してから評価用に固定する。
画面外投影は参照H由来の外挿として区別し、直接観測GTには数えない。

## 3. 新しい採用条件の案

「元画像内の全14点＋8inlier」を、**可視範囲の証拠と推定の安定性**へ置き換える。
閾値の値は参照確認前に調整しない。

1. Hのfinite/rank、投影poleなし、外周の凸性、画像との交差面積を確認する。
   画像外の点の座標は保持し、visibilityだけfalseにする。全点画像内は要求しない。
2. rawの有効点、重複・境界で打ち切られた点、Hへの支持点、fit点を分ける。
   最低4つの独立な対応を保ち、正規化DLT/Jacobianの悪条件性、支持点・支持線の広がりを調べる。
   点数が7でも十分拘束された部分コートを拾い、8点あっても同じ場所への集中を拒否する狙い。
3. 既存の双方向LINE残差と縦横の支持を維持し、平均だけで消えるbaseline等の不一致を
   線ごとに記録する。見えるはずの区間の不一致と、crop/画像外で未観測の区間を区別する。
4. KPの1点除外・線群の除外によるHの変動、別候補とのdisplacement/gapから外挿感度を測る。
   一部の証拠を外すと可視コートが大きく動く候補は拒否する。候補が1個しか残らない場合にも
   安定性の証拠を別途要求する。何も拘束されていない領域を小さい残差だけで合格にしない。
5. 領域の順位をraw点数優先から可視線一致・支持範囲・安定性へ変える。
   各cropのlocal値と原画像pxへ換算した値を併記し、同じ単位で比較する。

これは正誤を分ける仮説であり、効果の実証ではない。単にmin_inliersを下げる案は採らない。
full-frame LINEを共通座標で追加検証へ使う案も比較候補だが、隣接コートの線を
当該コートの誤差に数えるリスクがある。無条件の必須gateにはせず、対象線集合の定義を先に検証する。

## 4. 後処理の改善案

最初にraw evidenceと候補診断の保存を揃え、次に境界・重複への対応、最後に
共同fitと採否を変更する。境界からの距離だけで点を落とすと正しい端の交点も消えるため、
native heatmapのpeak形状・打ち切り・semantic点間の重複を合わせて候補の扱いを決める。
疑わしい点は「見えない点の正確な位置」としてseed/最適化へ与えず、理由を記録する。
ただし今回の代表例では既に除外済みなので、これだけを完了条件にしない。

現在の明示的なtrimming、最大8fit KP、画像外線のclipping、双方向LINEを再利用する。
支持の空間分布を考慮した候補多様性と外挿感度の確認を加え、近い解だけがseed上限を占める
場合を診断する。LINEの可視区間集合は評価中に都合よく消えないように固定して比較する。
十分な証拠がなければ理由付き失敗にし、KP-only、別clipのH、全frame共有へ切り替えない。

## 5. 評価と回帰

参照承認後、入力・checkpoint・frame 0・raw head・region集合を固定して、
旧後処理＋旧gate、旧後処理＋新gate、新後処理＋新gateを比較する。
276件と全候補の記録を残し、各条件の最終Hに対して参照から正誤ラベルを計算する。
旧Hのラベルを新Hへ流用しない。

- camera別に正推定採用／正推定拒否／誤推定採用／誤推定拒否を数える。
  正推定採用率、誤推定拒否率、全入力中の正しい採用率と失敗率も示す。
  参照の不確かな例は保留数を別掲し、成功した画像だけの平均へ隠さない。
- 参照で固定した可視点・未使用確認点・地面線の対応でmean/median/P95 pxを測る。
  予測visibilityで採点点を減らさない。画像外14点の誤差は推定参照に対する補助指標とし、
  直接観測の精度と区別する。cam3は同じ可視点でclip間MAD・P95散らばり、最大ずれを比較する。
- meiji_4camで設計した後に同じvideoの他clipで評価しても独立汎化の証拠にはならない。
  正誤のpx許容値・回帰の許容差・目標採用/拒否率を、最終比較前に人間と固定する。
- meiji_3camは既存manifestの57clip×3camera、tennis_clipは1clip×1cameraを
  前後比較する。既存manual参照の出自と現画像への適合も確認し、なければ同じ画像確認を行う。
  tennis_clipは単眼なので、3view以上を要求するscene完走率では評価しない。
- Courtの原TennisCourtDetector valは既存の順序とGTでKP-only／hybrid誤差、valid率、
  採否を比較する。既存の先頭300件の診断はsmokeとして再利用できるが、full-valの代用にしない。
  元sourceにはtestがない。video-disjoint registryのtestは学習への非混入を確認できた場合だけ使い、
  b863の学習runが不明なため未確認の独立testを名乗らない。合成の明示testも別表にする。
  旧checkpointと現行のdense教師schemaが違うため、非互換なLINE/SEG指標を混ぜない。
- raw headの不変性だけでは後処理の回帰検証にならない。Hの誤差・採否に加え、
  正しい部分観測のHが既存pinhole校正へ渡ること、除外cameraと理由、下流への影響を確認する。

ローカルにはmeijiのmanifest/mediaがある一方、既定の
`data/court_detection/tennis_court_detector-v1`と実画像評価registryは今回のcheckoutで未配置だった。
実装前に既存資産をhydrateし、版・split・学習来歴を照合する。今回は回帰推論を未実施。

## 6. 変更候補・テスト設計・判断点

実装で想定する主なファイルは以下。新しい共通部品は必要が判明した範囲だけに絞る。

| 対象 | 変更の目的 |
|---|---|
| `model_io/keypoint_decoder.py`・`model_io/contracts.py` | raw peakの境界・打ち切り・重複証拠を保持 |
| `geometry/confidence_homography.py` | seedのeligible集合と退化/悪条件診断 |
| `geometry/hybrid_homography.py`・`geometry/line_evidence.py` | 分布を考慮した候補、線区間の支持、安定性 |
| `inference/regions.py`・`inference/contracts.py` | 可視領域に基づく採否、候補順位、理由付き診断 |
| `src/tennis_scene/configs/pipeline.yaml`・`configuration.py`・`pipeline/components/court_kp.py` | 明示設定とschema/cache identityへの反映 |
| 対応するunit/integration、task/pipeline README、knowledge | 契約・受入記録。schemaを変える場合は全consumerを照合 |

テストは実装前に次の契約で設計する。今回のplanningで新規testは追加していない。

| 守る契約・検出したい不具合 | 既存nodeと増分案 |
|---|---|
| 正しい画面外投影の保持、crop→原画像座標 | `test_regions.py::test_crop_geometry_maps_to_full_image_without_relabeling_rasters`を再利用。`test_search_requires_geometry_and_raw_model_support`のoutside＝常に拒否を、支持十分な部分観測／不足の2契約へ変更 |
| 境界の正しい点を消さず、潰れた別semantic点を正確な観測にしない | `test_keypoint_decoder.py::test_decode_scales_normalized_peaks_to_original_pixels`を維持。native heatmapの正常端peak／打ち切りpeakと重複配置を追加 |
| 不使用KPが目的関数へ混入しない、4点でも退化を拒否 | `test_hybrid_homography.py::test_rejected_kp_have_exactly_zero_effect_on_objective`、`test_confidence_homography.py::test_degenerate_or_insufficient_input_has_no_fallback`を維持。ほぼ共線・偏在・独立な6点部分観測を追加 |
| 未観測と不一致の区別、潰れたコートや曖昧解の拒否 | `test_hybrid_homography.py::test_bidirectional_line_objective_penalizes_collapsed_court`と`test_cropped_segments_are_clipped_without_assuming_missing_lines`を拡張。平均支持が高くても可視baselineが誤る例、同scoreで異なるHの例を追加 |
| 明示失敗・単一設定・cacheへの反映 | `test_nonconvergent_joint_fit_has_no_kp_only_fallback`、`test_court_kp.py::test_failed_geometry_is_invisible_and_never_refit`を維持。採否理由、候補の出自、品質設定変更のidentity変化を追加 |

unitは非対称な既知Hと手計算の点、異常を注入したraw/LINEを使う。
integrationは人間確認済みの実画像・保存raw数枚で、正常部分観測、cam3端点集中、
cam0低支持の3fixtureを候補とする。巨大checkpointやGPUをunitへ入れない。
追加caseの時間目標はissue指定の各5秒以内。組合せを直積にせず、実測`--durations=0`で確認する。
実GPUの全件比較は受入用のqueue runとして分ける。

既存4fileの58testを6workerで実行し、全件成功、wall 13.26秒、最長call 1.45秒だった。
コマンドと結果はbundleの`baseline-tests.log`。これは現行baselineの検証であり、
未実装案の検証ではない。独立validatorは回数未指定につき0回。

主なリスクは、遠方の注釈精度、レンズ歪み、境界の本物の交点の過剰除外、
隣接コート・netによるbinary LINEの曖昧性、外挿感度確認によるCPU時間増加。
参考診断では1候補ずつの推論等11回で合計26.18秒だったが、モデルload/queue待ちを含む
benchmarkではなく、今後の処理時間予算を意味しない。

親に返す判断点は、(a)上の参照草案の人間による確認・修正、(b)正誤と回帰のpx許容・
目標採用/拒否率、(c)可視支持＋外挿安定性を中心とする実装方針。
現在のphase指示どおり、実装開始は親の具体指示を待つ。
#2191には将来、品質指標・採否理由・参照したclip/cropの契約を渡す。動画内共有処理は変更しない。

再現スクリプト、設定、全件のgate/目視分類、段階指標、未確認参照、入力の識別記録は
[run bundle](../../runs/run-court-acceptance-diagnosis-20261116/)に保存した。
大きいraw配列・原画像・全contact sheetは前記outputsに保持する。
スクリプトは実行時のoutputs配置で保存したものなので、再現時はbundleの3つの`.py`を
同じoutputsディレクトリへコピーし、worktreeから実行する。GPU captureは共有queueのhalf枠を使う。

summary.mdの「後処理は診断段階で、会場分離benchmark未確立」という判断を支持する記録であり、
checkpoint・採用方式・独立testの判断は変えない。このためsummaryへの影響はnoneとした。

再現補正（2026-11-16、PR #2200のKnowledge CI）: 元コマンドが指す未追跡の`capture_stages.py`が
空の保存patchに欠けていたため、bundleに既に保存されていた当時のscriptを追加diffとして補った。
元コマンド・script内容・測定値は変更していない。補正前後とsourceのhashは`repro-source-correction.json`に記録した。
