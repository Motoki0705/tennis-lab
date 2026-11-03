# pr-1673-1 実装前テスト設計

- 契約: HeatmapDecodingConfigの7 fieldとBallDetectionConfigの全fieldは必須constructor入力。数値と方式の既定値はpipeline.yamlのcompositionが所有し、標準pipeline・比較候補・テストへ明示的に渡す。
- 検出対象: Python既定値の再導入で型が監査から消える変更、composition overrideが比較へ届かない変更、値・型・復号数値の変化。
- 層: 既存のconfiguration契約unitのparametrizeへ両型を追加し、SOURCE_CONTRACT_DECLARATIONSとADAPTER_CONTRACTSに存在しrequired/composition-ownedであることを確認する。sceneのunitで非対称なtemperature/strideのoverrideを比較候補へ通す。既存の復号・pipeline・runner回帰を再利用する。
- 再利用node: tests/unit/utils/configuration/test_contracts.py::test_representative_domain_adapter_is_source_discovered、::test_adapter_fields_have_truthful_default_precedence_and_absence_policy、tests/unit/tennis_scene/test_configuration.py、tests/unit/tasks/ball_detection/inference/test_heatmap_decoding.py、tests/integration/tennis_scene/test_ball3d_heatmap_decoding.py。
- 予算: 新規は監査2 caseとcomposition接続1 case。GPU・モデル再推論は不要。pytest -n 0、--collect-only と --durations=0 --durations-min=0で件数と実測を保存する。src/utils/configurationの探索実装は変更しない。

実データ影響の判断: review対象HEADの7 defaultと現行YAMLの一致、compositionで作った候補設定とv4 selection.jsonの全設定の一致、数値演算bodyと固定入力・下流設定の不変を照合する。全clip再実行を省く場合は、この境界と根拠を記録し、実際に再計算したとは書かない。
