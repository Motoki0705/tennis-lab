# 実行前に固定する一覧

実行コード: `3bf09cf51a33422a5e4684298d957e37a3998b03`。この後の一覧だけのcommitはcheckout出自として別記する。
各行は同一bundle/GT checksum（execution-plan.json）と全view+3 LOVO。既存の選定条件をそのまま再生する。
GTを使う4 oracleは別の診断で、方式・組合せの順位へ含めない。LOESSなしも参考診断。

| 行 | 種類 | 設定の正本/照合元 | 出力先（integrated_comparison_v1/以下） |
|---|---|---|---|
| pipeline_baseline | baseline | foundation_v2 | results/pipeline_baseline |
| without_loess | reference_no_labels | foundation_v2 | results/without_loess |
| quadratic_logit_3 | method | heatmap_decoding_v4 | results/quadratic_logit_3 |
| selected_affine | method | heatmap_decoding_v4 | results/selected_affine |
| topk_multihypothesis | method | topk_multihypothesis_v2 | results/topk_multihypothesis |
| candidate_gmm_map_none | method | candidate_gmm_v2 | results/candidate_gmm_map_none |
| candidate_gmm_mean_none | method | candidate_gmm_v2 | results/candidate_gmm_mean_none |
| candidate_gmm_map_rts | method | candidate_gmm_v2 | results/candidate_gmm_map_rts |
| candidate_gmm_mean_rts | method | candidate_gmm_v2 | results/candidate_gmm_mean_rts |
| dense_selected | method | dense_likelihood_v2 | results/dense_selected |
| dense_temporal | method | dense_likelihood_v2 | results/dense_temporal |
| segment_physics_candidates | method | segment_physics_offset_v3 | results/segment_physics_candidates |
| segment_physics_heatmap | method | segment_physics_offset_v3 | results/segment_physics_heatmap |
| court_ball_ba | method | court_ball_ba_v2 | results/court_ball_ba |
| ba_cameras_baseline | combination | execution-plan.json の凍結合成設定 | results/ba_cameras_baseline |
| ba_cameras_topk | combination | execution-plan.json の凍結合成設定 | results/ba_cameras_topk |
| ba_cameras_dense | combination | execution-plan.json の凍結合成設定 | results/ba_cameras_dense |
| selected_affine_topk | combination | execution-plan.json の凍結合成設定 | results/selected_affine_topk |
| topk_loess | combination | execution-plan.json の凍結合成設定 | results/topk_loess |
| selected_affine_ba | combination | execution-plan.json の凍結合成設定 | results/selected_affine_ba |
| annotation_oracle | diagnostic_labels_used | error_decomposition_v2 | diagnostics/annotation_oracle |
| annotation_time_oracle | diagnostic_labels_used | error_decomposition_v2 | diagnostics/annotation_time_oracle |
| candidate1_oracle | diagnostic_labels_used | error_decomposition_v2 | diagnostics/candidate1_oracle |
| candidate8_oracle | diagnostic_labels_used | error_decomposition_v2 | diagnostics/candidate8_oracle |

合成対象外（別入力へ置換しない）:

- BA clock + top-K/dense: Existing top-K/dense kernels assume simultaneous frames. Applying nonzero clocks requires a new asynchronous likelihood/candidate model; camera-only rows explicitly use zero clocks.
- GMM K/support/failure-policy changes: Child next proposals require changing the model or retuning; not a composition of existing frozen parts.
- physics segment continuity/event support/dense convergence changes: Child proposals require new objectives or tuning, not existing component composition. Unconverged heatmap row remains in all tables.
- top-K null/motion cost changes: Requires retuning, prohibited. The existing selected-candidate + LOESS composition is included.
- LINE-based BA: Pinned bundle contains no raw LINE coordinates. Re-inference/new bundle is outside scope.
