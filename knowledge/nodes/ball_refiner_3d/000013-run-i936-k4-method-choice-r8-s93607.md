---
id: run-i936-k4-method-choice-r8-s93607
type: run
task: ball_refiner_3d
sequence: 13
recorded_at: '2026-09-30'
date: '2026-09-30'
title: '固定K=4標本: GT NLL・HDR・CPU費用から固定予算Hを採用'
provider: codex
status: done
issue: 936
config:
  sample_seed: 93607
  frames: 311
  components: 125
  workers: 4
  native_threads: 1
  hdr_samples: 8192
  calibration: provisional_context_free_ft_e13
  selected_method: fixed_hybrid
  voxel: {initial_cells: 8, levels: 4, refine_cells: 64, prior_sigmas: 5.0}
metrics:
  overall_nll_nat: 5.356963925808753
  weighted_nll_nat: 3.890161690412505
  weighted_hdr50: 0.5186633291614512
  weighted_hdr90: 0.7293316645807256
  weighted_hdr95: 0.7878748435544428
  weighted_mean_error_m: 4.529835956699627
  overall_mean_worker_seconds: 0.27086474528626786
  weighted_mean_worker_seconds: 0.27441338691093253
  failed_frames: 0
artifacts:
  run_dir: knowledge/runs/run-i936-k4-method-choice-r8-s93607
parents:
- run-i936-k4-audit-r7-s93607
- run-i936-provisional-degradation-r5-s936
relations:
- {to: run-i936-k4-audit-r7-s93607, rel: supersedes}
papers: []
tags: [cpu, probabilistic-triangulation, method-selection]
repro:
  commit: 7396614e3f5883775c9457af0060671935c99126
  command: 'bash knowledge/runs/run-i936-k4-method-choice-r8-s93607/repro.sh <absolute-new-output-root>; use the later evidence bundle with this implementation checkout'
---

**固定予算Hを生成器の既定に採用する。** 費用上限0.6秒/frame以下はAとHで、
Hは標本全体・停止データの層比率加重の両方で、AよりGT NLLと平均HDR較正誤差が小さい。
加重NLLは4.0293→3.8902、HDR較正誤差は13.17→11.72 percentage points。
Hの平均は標本全体0.271秒、加重0.274秒/frameで、全311件・全38,875成分を返し、float32保存でもSPDを保つ。

run 8 directiveに従い、方式選定を#936仕様の**GT 3D NLL・coverage・費用**へ戻した。
run 7の95%/欠損90%という数値収束目標は採用・生成のgateから外す。既存の収束flagは維持する。
Hを高精度積分や十分に較正された分布とは呼ばない。加重95% HDRは78.79%で、特に2camera層の誤差が大きい。
この暫定較正での相対選定と、最終#935・本学習・実データでの性能承認を区別する。

## 標本・予算・推論経路

[事前登録sample.json](../../runs/run-i936-k4-audit-r7-s93607/sample.json)は結果前の460bc987で固定。
SHA256は`df7753a4177ba5171971681064619cb06aaab3dac481409d7cf14781477dee7d`。
停止dev-r5の完成済みtrain9ラリー・3,995frameを、観測camera数0/1/2/3 × hit/bounce±5で8層化し、
seed93607で各層min(50,N)を抽出した311frameをそのまま使う。GTは採点だけに使い、再選別しない。
全9 source NPZのSHAを測定前・完了後に照合した。未完了ラリー・val/test・全640件を代表する保証はない。
frame相関があり、N=311を独立な標本として有意差を主張しない。

| 方式 | 結果を見る前に固定した予算・近似 |
|---|---|
| A | 正depthのGauss–Newton/Laplace、残差評価100回。最終の正depth点・Gauss–Newton共分散と、境界/tail/予算/line searchの診断を返す。予算停止をMAPと呼ばない |
| H | strictな正則A＋非正則productのvoxel。初期8³、4段、各段64cellを細分化、prior各軸±5σ。未細分化cell質量も保持し、cell幅²/12を共分散へ含める |
| ray | run 7と同じ12/20/32/48/64次、外側NLL/evidence 0.05nat・mean0.02m・covariance relative0.05、最低3段階 |
| adaptive_ray | run 7と同じray次数、5/7点cell則、cell cap64/128/256/512/1024、内側目標0.02/0.01/0.003/0.001/0.0003 |
| C | 全成分積ごとに8回、seed93608固定で2Dとprior中心を摂動して正depth最適化（各100評価）。積ごとに粒子群＋Scott KDEのmomentをGaussian化。重みは2D成分積で、geometry evidence補正なし |
| Q20 | strict A＋非正則rayを固定Hermite20次で1回積分。次数間の収束は測らない |

共通の空間priorはmean=(0,0,2)m、covariance=diag(36,144,9)m²。
#959暫定bank・2D成分・presence・camera・maskは変更しない。全方式で全125 productを保持する。
Cも少数の確率標本だけで成分積を選ぶのではなく全積を列挙し、各積の固定標本を使う。
seed変更・成分削除・covariance jitter・失敗時の別方式への自動切替はない。
Hの非正則dispatchは明示した単一の決定論的アルゴリズムで、成分ごとの理由を保存する。

比較実装/予算は7392f7f6で固定し、A/H/Cの結果を03624eab、生成器経路を7396614eで保存した。
実測commitと各sourceのSHAは各`*-results.json.gz`にある。途中の非数値変更は採点器のfloat32診断化と、
CheckedTriangulationへの`convergence_assessed`既定True追加。既存rayの成功308分布は
[run 7との全要素比較](../../runs/run-i936-k4-method-choice-r8-s93607/ray-regression.json)で完全一致した。adaptive rayの311分布も同様に完全一致した。

## 採点と全比較表

**[全体・8層・停止データ比率加重の比較表](../../runs/run-i936-k4-method-choice-r8-s93607/tables.md)** が数値表の正本。
[summary.json](../../runs/run-i936-k4-method-choice-r8-s93607/summary.json)には、存在・収束・float32・参考偏差も層別に保存した。

- NLLは全3D Gaussian mixtureのGTでの`−log q(x)`、自然対数、密度単位m⁻³。
- HDR mass αは`{x: log q(x) >= quantile_(1−α)(log q(X))}`、X~q。各frame8192標本、
  seed=`936080000+事前登録frame index`で全方式共通。GTを標本生成・閾値選択に使わない。
  50/90/95%を報告する。混合を単一の平均楕円体へ置き換えない。MCのmass推定SEは最大約0.55 percentage point/frame。
- 平均誤差は`||Σm wm μm − GT||₂`のframe平均(m)で、RMSEではない。
- HDR較正誤差は3水準の`|実測coverage−名目coverage|`の平均。coverageを常に大きいほど良いとは扱わない。
- overallは311frame等重み。weightedは各frameへ`元層の人口/その層の標本数`を付け、合計3,995へ正規化。
  元人口/標本は0-away=198/50、0-event=50/50、1-away=4/4、1-event=7/7、
  2-away=731/50、2-event=65/50、3-away=2552/50、3-event=388/50。
- 費用は4 worker/native thread1で並行実行した各frameのsolver wall秒。採点・adapter・I/O・起動は除外し、
  別に保存する。平均・p50・p95・maxを全群で報告。quantileは重み付き経験CDFの逆関数。
- ray/Q20は3件の数値失敗があり、N/fail欄と失敗の重みを残す。その行のNLL/HDR/平均誤差は**成功例に条件付き**で、
  全311件の有限NLLを主張しない。失敗例も時間・分母・JSONに残し、選定から黙って消さない。

## 選定理由・適用限界

費用適格はAとH。HはAに対してoverall NLL5.5264→5.3570、加重NLL4.0293→3.8902、
overall HDR較正誤差18.53→17.03pp、加重13.17→11.72ppへ改善する。
各HDR水準の名目値からの乖離も、この2集計ではHが小さい。
加重混合平均誤差も4.752→4.530m。層ごとの完全な優位ではなく、3camera/event層等ではAが僅かに良い。
条件ごとの方式変更やGTに応じたdispatchはせず、全frameに同じHを採用する。

adaptive rayは加重NLL3.8972・95% coverage80.07%でHに近いが、加重平均5.557秒/frame。
Hの約20倍の費用を払う理由を今回のGT指標からは得られない。
Cは90/95% coverageを改善するが、NLL4.6548、平均誤差5.768m、1.235秒/frame。
Q20も1.039秒/frameで上限を超え、3件の処理失敗が残る。固定したC実装の結果から全sampling法を否定しない。

既存の空camera集合のprior単体をGTで採点すると、加重NLL9.0862、平均誤差10.030m。
Hはこの無情報成分より密度・平均位置に情報を持つ。一方、HDRの過小被覆は残る。
[prior参考値と入力SHA](../../runs/run-i936-k4-method-choice-r8-s93607/verification.json)は採用候補を追加したものではない。
#959は旧detector・文脈なしpilotであり、長gap/camera相関/負例/新epoch-9 detectorは未較正。
Hの選定は#935最終出力の改善や実Meijiでの性能を保証しない。

## 存在・数値診断

各camera集合Sの総質量は全方式で独立Bernoulliの`∏pᵢ∏(1−pᵢ)`を維持する。
Hの最大質量誤差は4.44e−16、全体prior-only質量平均0.0001551。空Sは指定Gaussian priorそのもの。
観測camera数0は球やcamera内amodal存在の否定ではなく、#935が補完した分布も入力にする。
`prior_only_probability`を球の不存在確率に読み替えない。全311件に合成3D GTがあり、球自体の不存在の精度は測れない。
933 camera-frame中120はout_of_frame。記録したamodal Brierは劣化recipeの記述値で、
画面外logitを既知maskから−4へ設定した暫定仮定を含むため、#935モデルの較正性能と解釈しない。

rayは142/311（加重36.34%）、adaptiveは220/311（加重60.00%）が従来の数値判定を通過。
全欠損100件では44/74件。Aは全productのoptimizer停止判定が通ったframeが202、Cは0だが、
どちらも最終の全分布を採点した。これは方式間で同じ意味の収束率ではなく、採用のgateにもしていない。
H/Q20には次数間の収束推定はない。Hは生成時も**未評価**を明示し、false flag/0 deltaを収束の証拠に使わせない。

A/Cはfloat64では全311件成功するが、float32へ丸めると全成分SPDを保てないframeが100/79件あった。
初期Aではこの保存検査が採点を止めたため、その記録を`A-export-gate-results.json.gz`に残し、
採点器だけを修正して同じ全311件を再測定した。数値経路・budget・seedは変更していない。
Hの選定根拠はNLL/coverage/費用であり、このfloat32診断を数値収束gateの代わりにはしない。
Hは全38,875成分についてfloat32 SPDを確認した。float32→float64読込後のGT NLL差は
最大7.99e−6nat、混合平均差は最大7.84e−7m。float32で0になった599重みも成分配列を保持した。

adaptiveが収束した220frameでの参考偏差（GTの代用にはしない）:

| 方式 | 比較数 | GT NLL絶対差 mean/p95/max nat | 混合平均差 mean/p95/max m |
|---|---:|---|---|
| A | 220 | 0.2073 / 1.3717 / 3.9299 | 0.6389 / 6.6361 / 10.2527 |
| H | 220 | 0.04179 / 0.25977 / 0.92505 | 0.08641 / 0.33466 / 4.60573 |
| ray | 218 | 0.000112 / 0.000552 / 0.003126 | 0.000210 / 0.000386 / 0.016761 |
| C | 220 | 1.2354 / 3.6055 / 8.8135 | 2.6437 / 10.5195 / 26.8369 |
| Q20 | 218 | 0.000227 / 0.000681 / 0.008349 | 0.000878 / 0.000883 / 0.073621 |

Hをadaptiveと同一の分布とは扱わない。有限box/粗いcell/各productのGaussian要約には近似誤差があり、
平均位置の最大偏差4.61mも残す。収束した参考方式にもLaplace/moment近似・共通mode見落としの限界がある。

## 生成器・96ラリーの提案

`dataset_plan.yaml`は上記Hの予算を明示し、generatorはutilsの`triangulate_conditioning`を呼ぶ。
この関数は観測GMM/camera/priorだけを受け、学習用生成と将来のpipelineが共有できる。
`integration_convergence_assessed`と未評価件数を追加し、readerは固定予算で偽の収束flagを立てたdataを拒否する。
既存ray/adaptive dataは従来の達成差分・flagを保持して読める。pipelineへの接続は今回行っていない。

**96件は提案だけで、開始していない。** CPU4 process/native thread1、train/val/test=64/16/16、
最大512frame、GPUなし。物理simは停止9件の平均1.766秒/rally、その他0.183秒/rallyを加算した。

| ラリー長仮定 | 96件 wall外挿 | 20%余裕込み | NPZ容量外挿 |
|---|---:|---:|---:|
| 450frame/rally | 0.841 h | 1.009 h | 0.271 GB |
| 最大512frame/rally | 0.955 h | 1.145 h | 0.308 GB |

orchestratorへの申請は**90分枠・CPU4・RAM概算6GB・disk予約0.75GB**。
容量は旧停止データの6,275 bytes/frameからの外挿で、JSON・新規診断・split差の余裕を別途取る。
新しい固有出力directoryへ生成し、旧dev/frozen worktreeを保持する。GTや結果によるseed再試行・成分削除はしない。
#959の`calibration.json`を`--calibration-report`の単一入力として使う。
report SHAは`7a2ea393b0a4c4c62c9f5903b577af9a64bb7fa9a5cc9b5f0fb4a31378ad360d`、
bank SHAは`aea209600cfe218e239612cfaa828706b7ffabedebc9291ea0634acce31c482c`。
新#935 reportが届いたら同じ入力境界で差し替え、異なる較正条件として監査する。

[計算式・資源提案・入力path](../../runs/run-i936-k4-method-choice-r8-s93607/selection-and-cost.json)を保存した。
640件の20%余裕込み外挿は450frameなら6.72h、512frameなら7.64hだが、**最終#935出力を待つ**。
これは起動申請ではない。現planの古いpilot容量概算1.5GBも、全量を予定する際に再見積もる必要がある。
10分/59.94fpsのsolverだけなら4 CPUで約41分という算術外挿で、pipeline実測・統合を意味しない。

## 検証と残る作業

比較/既知Gaussian/positive depth/HDR等40 tests成功。その後、geometry/refiner/configurationの
**717 tests成功（162.90秒）**。生成器の2frame解析的fixtureでNPZ保存→reader、全125成分、
未評価flag、偽の収束flag拒否も検査した。96件生成の実測や本学習ではない。
ruff/mypy/pre-commit成功。commit hookでtestの型注釈1件を修正した。
前回38843ed4のPython CIも6,556 passed / 133 skippedで成功（保存JSON）。
7396614eのknowledge CIとPython CI内の同じrepro検査が、独自ROOT変数表記を解決できず失敗したため、
同じ作業rootを表す`$PWD`へ直し、repro検証は0 missing/0 unverifiableを確認した。
そのPython CIでは他の6,566 testsは成功、133 skipped。修正後の該当e2e検査3件も成功（10.70秒）。
最新headのCI・レビューガイドはPR #969のrun8コメントに記録する。

全frame記録、全component配列、使用budget、source SHA、GT採点閾値、全層集計をbundleへ保存した。
新規成果物の容量は`disk.json`を参照。GPU/queue、96/640件生成、実Meiji球評価、pipeline接続は行っていない。
第1 acceptanceの比較・選定・共通実装/unit testをこのK=4条件で更新した。
第2項full、第3項の#929/回帰との優位、第4項pipeline/imports撤廃は未完了である。
