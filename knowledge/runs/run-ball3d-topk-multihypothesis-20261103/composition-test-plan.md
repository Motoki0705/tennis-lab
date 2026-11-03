# Composition所有権の修正前テスト設計

対象はPR #1672への2026-11-04追加指摘。`TopKConfig`の実際の16項目、
評価adapterの`settings` factoryと`min_frames`、今回追加した関数引数の既定値を確認する。
方式・閾値・評価分割は変えない。

| 守る契約 / 検出する故障 | 層・既存node / 追加理由 | 予算 |
|---|---|---|
| TopKの全設定はcomposition所有。既定値を1項目戻して宣言・nested探索から消える故障を検出 | `tests/unit/utils/configuration/test_contracts.py::test_topk_settings_are_discovered_as_required_composition_owned_fields`を1件追加。既存のcatalog parityでは探索対象からの脱落を検出しない | 既存catalog importを共有、callは1秒以内目標 |
| YAMLの必須項目削除を拒否し、CLI overrideの0を保持 | `tests/unit/tennis_scene/test_configuration.py::test_ball_method_and_all_topk_settings_come_from_composition`を維持し、必須topk scalar欠落の1件を追加 | Hydra composeは各case1回、追加case5秒以内目標 |
| productionと評価adapterが同じ明示設定で同じ3D・候補を返す。camera/frame交換、LOVOのheld-out情報流入を検出 | 既存`tests/integration/tennis_scene/test_ball3d_topk.py`の2件を再利用。非対称3camera/18frameのfixtureを維持 | 各case5秒目標、外部資産・推論不要 |
| 既定pipelineのmain配列・float32 residual契約と、時系列gap/turn/候補選択を維持 | 上記結合テストと`tests/unit/tennis_scene/ball_association/test_topk.py`を再利用。テスト設定も実compositionから明示的に渡す | CPU、pytest -n 0、既存suiteの範囲 |
| YAMLの16値・min_framesが修正前Python値と同じ。固定bundleの8run全配列と診断がimmutable v2と一致 | 旧HEAD AST literal / Hydra composition比較、既存`verify_ci_predictions.py`によるbaseline+方式の全view+3LOVO再生 | 約1分/635MiB、単独実行、GT・動画・GPU不要 |

設定監査全体と関連suiteは`measure_tests.py`で`--collect-only`、
`--durations=0 --durations-min=0`を記録する。既存receiptは上書きしない。
`src/utils/configuration`自体は変更しない。ruff/mypy/knowledge validatorも実行する。
