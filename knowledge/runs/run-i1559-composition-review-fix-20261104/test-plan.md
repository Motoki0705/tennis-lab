# pr-1673-1 実装前テスト設計

- 契約: HeatmapDecodingConfigの7 fieldとBallDetectionConfigの全fieldは必須constructor入力。数値と方式の既定値はpipeline.yamlのcompositionが所有し、標準pipeline・比較候補・テストへ明示的に渡す。
- 検出対象: Python既定値の再導入で型が監査から消える変更、composition overrideが比較へ届かない変更、値・型・復号数値の変化。
- 層: 既存のconfiguration契約unitのparametrizeへ両型を追加し、SOURCE_CONTRACT_DECLARATIONSとADAPTER_CONTRACTSに存在しrequired/composition-ownedであることを確認する。sceneのunitで非対称なtemperature/strideのoverrideを比較候補へ通す。既存の復号・pipeline・runner回帰を再利用する。
- 再利用node: tests/unit/utils/configuration/test_contracts.py::test_representative_domain_adapter_is_source_discovered、::test_adapter_fields_have_truthful_default_precedence_and_absence_policy、tests/unit/tennis_scene/test_configuration.py、tests/unit/tasks/ball_detection/inference/test_heatmap_decoding.py、tests/integration/tennis_scene/test_ball3d_heatmap_decoding.py。
- 予算: 新規は監査2 caseとcomposition接続1 case。GPU・モデル再推論は不要。pytest -n 0、--collect-only と --durations=0 --durations-min=0で件数と実測を保存する。src/utils/configurationの探索実装は変更しない。

実データ影響の判断: review対象HEADの7 defaultと現行YAMLの一致、compositionで作った候補設定とv4 selection.jsonの全設定の一致、数値演算bodyと固定入力・下流設定の不変を照合する。全clip再実行を省く場合は、この境界と根拠を記録し、実際に再計算したとは書かない。

## CIで検出したDINO unitの状態分離（修正前設計）

- 契約: 仮のDINO repository/extensionを使う既存unitは、先行する実DINO importの有無に依存しない。テスト自身が作るstaleな拡張参照は引き続き拒否する。
- 不具合: 初回CIのlane 9で、親packageだけを初期化したテストに先行importのfunctions子moduleが残り、正常系の最初のloadが拒否された。productionの拒否挙動は正しい。
- 層・再利用node: tests/unit/submodules/models/dino/test_extension.py::test_runtime_import_uses_selected_module_and_rejects_cached_mismatch の準備処理だけを修正する。新規case・production変更・GPUは不要。
- 検出方法: functions子moduleに別extensionを持つ先行状態を明示的に注入し、同じ既存nodeが修正前にCIと同じRuntimeError、修正後に成功することを確かめる。node後半の意図したstale参照拒否assertも保持する。
- 予算: 既存DINO extension unit全体を-n 0でcollect/実行し、件数と全phase時間を記録する。プローブは1caseずつ、全clipや設定監査一式の再実行は不要（変更はこのunitの準備のみ）。
