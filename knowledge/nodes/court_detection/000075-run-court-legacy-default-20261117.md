---
id: run-court-legacy-default-20261117
type: run
task: court_detection
sequence: 75
recorded_at: '2026-11-17'
title: clip既定の旧処理復元と明示選択部品の保持
issue: 2190
provider: codex
date: '2026-11-17'
status: done
summary_impact: updated
config:
  agreement_revision: 4
  support_method: legacy
  acceptance_method: legacy
  refinement_method: model
  GPU_used: false
  training_queue_jobs_added: 0
  new_neural_raw_acquisition: false
  cpu_workers: 10
repro:
  commit: 082b085202b124cf72c1e551cd42c4c7fe883c27
  command: CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
    .venv/bin/python knowledge/runs/run-court-legacy-default-20261117/replay.py --workers
    10
metrics:
  acceptance_revision_4_met: true
  images: 3545
  regions: 8530
  mismatch_images: 0
  mismatch_regions: 0
  cpu_wall_seconds: 4353.849037672975
  pytest_cpu_passed: 245
  pytest_cuda_skipped: 1
  pytest_seconds: 49.3
  pytest_max_case_seconds: 4.8
  added_test_cases: 2
  mutation_checks: 2
  meiji_3cam_regions: 2109
  meiji_3cam_accepted: 165
  meiji_3cam_any_h: 171
  meiji_3cam_all_exact: 171
  meiji_3cam_images: 171
  meiji_4cam_regions: 3312
  meiji_4cam_accepted: 152
  meiji_4cam_any_h: 276
  meiji_4cam_all_exact: 276
  meiji_4cam_images: 276
  synthetic_court_test_regions: 886
  synthetic_court_test_accepted: 30
  synthetic_court_test_any_h: 579
  synthetic_court_test_all_exact: 886
  synthetic_court_test_images: 886
  tennis_clip_regions: 12
  tennis_clip_accepted: 1
  tennis_clip_any_h: 1
  tennis_clip_all_exact: 1
  tennis_clip_images: 1
  tennis_court_detector_val_regions: 2211
  tennis_court_detector_val_accepted: 2153
  tennis_court_detector_val_any_h: 2211
  tennis_court_detector_val_all_exact: 2211
  tennis_court_detector_val_images: 2211
  pytest_related_passed: 211
  pytest_config_passed: 64
  pytest_overlap_cases: 30
  pytest_related_seconds: 37.17
  pytest_config_seconds: 12.13
artifacts:
  run_dir: knowledge/runs/run-court-legacy-default-20261117
  output_dir: .worktree-outputs/court-acceptance-postprocess/tennis_scene/evaluate/court-legacy-default-20261117
parents:
- run-court-semantic-support-20261117
relations: []
papers: []
tags:
- legacy-compatibility
- CPU-replay
- explicit-opt-in
---

[revision 4の合意](https://github.com/Motoki0705/tennis-lab/issues/2150#issuecomment-6933831373)で、
1frameの目標は#2191の動画共有モードへ移した。#2190の受入は、clip単位の既定が旧処理と一致し、
残した部品のテストが通ることへ絞った。run 69〜74の成功・失敗・承認記録は変更しない。
過去の単frame誤採用・回帰を解決したという意味での合格にはしない。

## 既定動作と部品の範囲

- `support_method=legacy`: 元のconfidence順位、支持集合を再選択する反復fit、LINE目的関数。
  境界打切り・重複相関群・空間分布優先・初期可視LINEの保持はこの既定では使わない。
- `acceptance_method=legacy`: 全14点の画像/content内投影、凸性、元の面積比、8点以上のinlier。
  float32投影を使う判定順序も戻し、inlier数→confidence和→提案順で選ぶ。
- `refinement_method=model`: 既定では品質数値計算もRGB fitも実行しない。
  探索無効時は元のfull-frame hybrid結果と画像内validityを渡す。
  `court_legacy_geometry_v1`はgeometry成立だけで、領域採否は候補診断に分離する。
- 支持点・採否・精密化の方式と数値既定はdataclass/pipeline YAMLで一致する。
  領域探索の`enabled`は汎用dataclass=false、pipeline=trueという旧仕様を保つ。方式・全設定・実装fingerprintはcache identityに含まれる。

|保持する部品|用途・明示方法|検証|
|---|---|---|
|境界の片側残差・相関群、固定集合solver|`fixed_subsets` / `candidate_subsets`を指定。各solveの実際の集合・H・残差・停止理由を保持|observations、既知H＋外れ値、集合不整合からの候補探索、失敗時fallback禁止|
|モデル品質と不確かさ|`acceptance_method=quality`で使用。schema/単位/理由/scoreを保持|部分観測、独立線不足、不一致、実raw integration|
|RGB stripeの中心測定・線残差・coverage|`refinement_method=rgb_paint`＋qualityを指定。`semantic_consensus_v2`のみ保持|既知Hからの人工塗線、ネット、遮蔽、画像外部分、意味曖昧性、支持喪失、外挿不確かさ|
|原HとRGB Hを分ける品質schema|モデル内部一致と物理白線残差を分け、不合格なら採用HはNone|region/scene componentの座標・診断・ゼロmask・永続化|

削除: PR中の中間比較専用`adaptive`（境界点除外＋空間順位の再選択反復）と、
全coarse stripeを必須にした`paired_edges_v1`のproduction分岐。
旧処理への暗黙切替は追加せず、削除した方式名は設定エラーにする。
当時の方式・結果はrun 69〜74とそこに記したcommitで再現する。
複数frameによる参照fit scriptはknowledgeに残し、srcへ移さない。

## 比較の根拠

[再現手順](../../runs/run-court-legacy-default-20261117/reproduce.md)と
[旧ソースの来歴](../../runs/run-court-legacy-default-20261117/baseline-source.json)に従う。
分岐元の旧コードと同一のsolverを保存LINE配列で実行し、新既定と対比較する。
同じ入力のH・投影・最終支持mask・残差はdtype/shape/bytes完全一致で検査する。
status・候補件数・refinement件数・score gap、全領域の採否・confidence・面積・凸性・内包判定、
選択領域も比較する。基準値の緩和、GT参照による調整、別Hへのfallbackは行わない。

Court val/syntheticは凍結rawに含まれるfull-frame、sceneは全proposalを比較する。
各画像の最初の全画面候補は領域探索なしの出力も照合する。
これは既定の同等性の受入であり、GPUニューラルforwardとraw再生成は含めない。

初回run 70のLINE距離場が未保存だったため、syntheticの旧成立数573と
run 73保存LINEでの旧replay579には数値環境差がある。その履歴は書き換えず、
今回の入力を揃えた旧ソースとの比較から実装差を判定する。

## テストの設計と限界

既存`test_line_corrects_biased_kp_and_excludes_confident_outliers`をlegacy/候補方式の2条件へ拡張し、
既知Hの回復と実際の支持集合を両経路で守る。
`test_gate_excludes_geometric_and_line_outliers_and_never_uses_all_points`に元のconfidence順・同点順を加えた。
新規region testは外側の点・7点支持・支持数優先・confidence優先・完全同点順を1つの固定候補列で検査し、
既定で新品質/RGBが呼ばれたら失敗する。full-frameの部分観測validityも比較する。
既存の新品質/RGB testは明示profileへ変更し、診断・不合格伝播の契約を維持した。
config testは方式ごとのcache差、削除方式とlegacy+RGBの拒否を確認する。
今回のtest増分はregion 1ケース＋既知H legacy 1ケース。数値検証nodeの削除なし。
旧設定キーの拒否2ケースはrevision 4で契約が変わったため、同数の受理・値域検査ケースに置き換えた。

受入目標を移したので、参照とRGB抽出器の共有による循環性、既存sceneのGT不足、
syntheticと学習splitの重複未確認、ネット上端やbaseline IDの誤対応は解消を主張しない。
これらは明示方式を使う#2191の判断材料としてrun 71〜74に残る。
GPU lockの修復・迂回はしておらず、実機CUDA確認は未実施。
再学習・checkpoint変更はなく、TensorBoard曲線の対象外。

関連テスト最終実行はCPU 2 workerで211 passed / 1 skipped、37.17秒。
CUDA専用nodeは`CUDA_VISIBLE_DEVICES=''`によりskipした。最長callは4.80秒、
追加したlegacy既知Hケースは1.84秒だった。CPU replay 10 processとpytest 6 workerを
重ねた先行実行では最大8.07秒だったため、2 workerで計測を確定した。
2つのin-memory mutation（既定をqualityへ変更、既定の支持点順位を空間分布優先へ変更）は
それぞれ対応するassertで検出し、復元後のテストも成功した。
pre-commitのruff/mypyは実装commitで成功。詳細はrun bundleのlogsとmutations.jsonへ保存する。

## 受入結果

**revision 4の受入を満たした。** 全3545画像/8530領域でH・投影・最終支持mask・残差、
solver状態と採否・選択領域が完全一致した。許容差を使う比較ではなく、配列のdtype/shape/bytesを照合した。
CPU 10 processで4353.85秒（72.56分）。途中で入力・数値ソースを変更していない。
YAMLとdataclassの方式・数値既定の一致（`enabled`を除く）、およびrun 73で保存した旧採否・選択領域との一致も全件確認した。

|入力|画像|領域|H成立（旧=既定）|採用（旧=既定）|全比較完全一致|
|---|---:|---:|---:|---:|---:|
|meiji_3cam|171|2109|171|165|171|
|meiji_4cam|276|3312|276|152|276|
|synthetic_court_test|886|886|579|30|886|
|tennis_clip|1|12|1|1|1|
|tennis_court_detector_val|2211|2211|2211|2153|2211|

Court valの114件喪失は旧処理の問題ではなく、以前のPR変更によるものだった。
既定を戻した結果、全2211件で旧Hと完全一致し、喪失は0件。syntheticも旧Hの喪失・獲得はともに0件。
meiji3の採用は165件へ戻った。これは既定の互換性の達成であり、単frameの推定精度目標の達成ではない。
同一H・maskなので凍結入力でのGT評価結果も変わらないが、新しいGT精度改善を主張しない。

全raw 8530・保存LINE 8530・RGB 3545はrun 73のinventoryのサイズ/hashと一致した。
[結果集計](../../runs/run-court-legacy-default-20261117/summary.json)、
[入力確認](../../runs/run-court-legacy-default-20261117/input-integrity.json)、
[全件H/診断のinventory](../../runs/run-court-legacy-default-20261117/artifact-inventories.json)と
[検出力確認](../../runs/run-court-legacy-default-20261117/mutations.json)を保持する。
CIの最終HEAD結果はPR #2200へ記録する。merge・issue close・#2191実装は親側の後続作業。

初回push `2ea001b59` のCIでは、`test_configuration.py`の
`test_removed_court_acceptance_keys_fail_explicitly` 2ケースが失敗した。
復元したキーを削除済みと期待する古い契約が残っていたためで、このworkerのテスト更新漏れだった。
ローカルでも2 failedを再現し、`test_legacy_court_acceptance_keys_accept_overrides_and_validate_ranges`へ
置換して、設定キーの受理・cache反映・値域外拒否を確認した。
両設定test fileは64 passed、12.13秒。数値実装と全件replay結果は変更していない。
[CI失敗と修正の記録](../../runs/run-court-legacy-default-20261117/ci-first-attempt.json)を保持する。

ローカル成功nodeの和集合は245ケース（関連211＋設定64−重複30）、CUDA skipは1ケース。
上記2 suiteの実行時間の和は49.30秒で、最長case 4.80秒。
