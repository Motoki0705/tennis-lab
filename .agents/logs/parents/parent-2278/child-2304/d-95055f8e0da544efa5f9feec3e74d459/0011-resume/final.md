[PR #2342](https://github.com/Motoki0705/tennis-lab/pull/2342) のmajor 4件を修正し、`codex/trt-dino` に `1abf84c4c` をpushしました。

1. **ncuの終了処理** — タイムアウト時はプロファイラと子孫を停止し、子が残っていないことを確認してから失敗を返します。v6のCSVがqueue終了確定後に更新された事実も記録しました。[試行記録](https://github.com/Motoki0705/tennis-lab/blob/1abf84c4ca62eb225a920dc0218b2933016aa0e3/knowledge/runs/run-i2304-dino-tensorrt/attempts.json)
2. **条件2の精度低下** — 原因は未確認と明記しました。条件1ではMSDA全体がFP32で、条件2ではMSDA方式とSwinを含むONNX全体の精度配分が同時に変わります。同一精度でのexport前後の照合と、MSDAだけをFP32に戻す対照は未実施です。[runノード](https://github.com/Motoki0705/tennis-lab/blob/1abf84c4ca62eb225a920dc0218b2933016aa0e3/knowledge/nodes/tennis_scene/000078-run-i2304-dino-tensorrt.md)
3. **3a・3bのPTQ適用範囲** — 両条件の実測ONNXで、指定外のAdd 36件、LayerNormalization 5件、Where 1件がDQ出力を直接受けていました。そのグラフでの実測値として記録し、prepare/buildに共通の `verify_qdq_scope` を適用しました。保存済みの旧グラフは両方ともGPUビルド前に終了値1で拒否されました。[metrics](https://github.com/Motoki0705/tennis-lab/blob/1abf84c4ca62eb225a920dc0218b2933016aa0e3/knowledge/runs/run-i2304-dino-tensorrt/metrics.json)
4. **プラグインの読込と検査** — 照合済みの変換記録から方式を渡し、プラグイン版だけを初期化します。実行層種別と入出力精度も照合し、同名のFP32 Castへの差し替えを拒否しました。[DINO回帰テスト](https://github.com/Motoki0705/tennis-lab/blob/1abf84c4ca62eb225a920dc0218b2933016aa0e3/knowledge/runs/run-i2304-dino-tensorrt/test_dino_protocol.py)

追加テストは4本、パラメータ展開込みで7件です。

- `test_ncu_timeout_stops_orphaned_writer_before_returning`
- `test_dino_build_rejects_qdq_feeding_unselected_residual`
- `test_verified_receipt_controls_plugin_initialization`
- `test_dino_plugin_build_rejects_wrong_layer_or_formats`

関連CPUテストは **140件成功（8.66秒）**、#2335との互換性テストは **5件成功**。Ruffとknowledge検証は通過し、pre-commitの2 hookは対象パス外でskipされました。GPUの再計測は未実施で、新しいPTQ設定での再計測は #2278 の判断事項です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2342,"head":"1abf84c4ca62eb225a920dc0218b2933016aa0e3","resume_on_success":false}}