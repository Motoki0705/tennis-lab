# PR #1672 CI修正の事前設計

対象: 87ed8239524925603f6473064937782bc24a812d / CI run 41603205543。

| 契約 | 検出する故障 | 層・再利用node | コスト |
|---|---|---|---|
| composition-ownedなruntime adapterはPythonのdefault/default_factoryを持たず、既定値もCLI overrideもcompositionから来る | ball_methodの修正後にball_topk等の違反を残す、明示した値をPython既定へ戻す | tests/unit/utils/configuration/全体を再利用、既存tennis_scene/test_configuration.pyへcomposition契約を追加 | -n 0、収集とphase時間を記録。追加1case <5秒 |
| 既定経路のpositions/valid/inliers/uv/rejectionはmain固定配列と完全一致。reprojectionだけfloat32の丸め水準で比較 | 他配列まで許容誤差を緩める、画素単位で意味のある残差差を通す | integration/test_ball3d_topk.py::test_default_pipeline_matches_main_arraysを拡張 | 既存12frame、case <5秒。solverソースの同一性も別記録 |
| float64のfitからpositionsとresidualを別々にfloat32へ格納する経路と機械依存の下位bitを確認 | 保存済みfloat32位置から残差を再計算したと誤解する | CPU機能/BLAS設定を変えた小fixtureの直列probe。CIの差と許容値をJSONへ記録 | 一度に1 Python process。GPU/動画なし |
| 設定変更で固定clip_000の予測は変わらない | 実際に別のTopKConfigをpipelineへ渡す | compositionの全fieldをv2 frozen-configと照合。必要なら基盤runnerでCPU再実行し保存済み4foldと配列比較 | v2 directoryは不変、新規receipt directory。heavy処理は直列 |

追加の設定dataclass/fieldは、監査のsource discoveryとtransitive typed-adapter検査に従って全体を確認する。既存の計算用configやmethodのPython API defaultを設定adapterのfallbackと混同しない。DINO隔離失敗が確認された場合だけa8982a7e0の指定ファイル差分をそのまま取り込む。

検証はconfiguration監査全体、synthetic visualization/contracts、上記integration、既存top-K関連106caseを -n 0 で実行。collect-onlyとdurations=0/durations-min=0を保存し、ruff/mypy/knowledge validatorを通す。push後は対象headのCI全件成功まで確認する。
