# #1560 実装前のテスト設計

前提: #1554 revision 6、基盤92e199b。GTは方法入力へ渡さず、tuningは[0,505)、主評価[505,1010)。CPUのみ、pytest -n 0、heavy replay/mediaは直列。

| 契約 | 検出したい故障 | 層 / node予定 | コスト上限 |
|---|---|---|---|
| gate前のvalidなK<=8候補をscoreで切らず、view間再投影と時間で選択。camera順に選択index/実pixelを返す | argmax固定、camera/frame軸交換、低score切捨て、1viewの長い張り付き | unit/tennis_scene/ball_association/test_topk.py::test_multiview_distractor_and_sticking_recover_asymmetric_trajectory | 3camera x 18frame x 2候補、1case <5s |
| 2view未満はzero/false/-1、短い欠測は時間差付きでリンク、長い欠測は履歴をリセット。補間出力しない | 欠測を観測化、gap時間を1frameとして計算、無制限リンク | 同 ::test_missing_limit_reacquires_without_fabricating_points | 小fixture <5s |
| 急な速度方向変化に有限の加速度コスト。実イベント/GTを使わない | 純二乗加速度でbounce/hitを棄却 | 同 ::test_turn_cost_keeps_bounce_and_hit | 小fixture <5s |
| 不正K/探索上限/非有限設定/入力軸を例外拒否 | 設定の無視、欠落evidenceへのfallback | 同 ::test_invalid_configuration_or_axes_fail、integrationの欠落evidence拒否 | 合計<=12展開、各<1s目標 |
| 既定triangulationの数値・mask/reasons/uvはmainと完全一致 | optional方式を既定化、既定入口/重み変更 | unit/pipeline既存test_ball_points.pyを再利用、新integration::test_default_pipeline_matches_main_arrays | 非対称12frame、mainの固定期待配列fixture。<5s |
| config->definition->assembler->方法->実codecで候補選択を保持 | ball_pointsで候補消失、依存先間違い、camera軸交換、保存で診断欠落 | integration/tennis_scene/test_ball3d_topk.py | 一度の小fixture、各<5s |
| LOVOは保持viewのevidenceのみ。共通runnerと保存/採点へ接続 | heldout view漏洩、候補index保存漏れ | 上のintegration + 再利用test_scoring.py::test_method_runner_excludes_labels_reference_and_heldout_camera | 四fold x 小fixture、<5s |
| 候補一覧と選択した候補を実videoに重ねる。既定描画を維持 | 3D再投影だけ描く、別cameraのindexを描く | visualizationの軽量overlay unit + 既存integration/test_ball3d_media.py | overlay <1s、実media再利用1件 |

既存node: test_baseline.py::test_without_loess_keeps_acceptance_but_uses_original_detector_coordinates、test_ball_points.py::test_filled_and_rejected_points_cannot_supply_the_second_real_view、test_ball3d_bundle.py::test_real_component_codec_bundle_and_cpu_runner_roundtrip を再利用。既存nodeの削除/統合はしない。

実装案: 全candidate pairからDLT仮説を生成し、追加viewの候補を幾何/scoreで対応。pair-onlyも残し、上位Hに剪定。速度と打切り加速度、欠測/restartコストを持つ有限beam動的計画法。これは全組合せに対する厳密なViterbiではなく、計算量を制限した探索。短い欠測も出力は欠測のまま。閾値調整は前半の共通scorerだけで行い、設定を固定してから後半を採点する。

計測: collect-only と durations=0/durations-min=0、junit XMLから全node/stage実測を記録。1case>5s/setup>1sなら原因と層の必要性をPRに記す。代表故障（candidate/view軸の取り違え又は時間項無効化）を一時注入し、意図した数値assertで失敗してから復元する。
