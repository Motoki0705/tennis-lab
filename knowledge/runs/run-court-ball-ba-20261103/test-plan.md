# 実装前のテスト設計 (#1567)

2026-11-03。親 #1554 revision 6、基盤 #1669 merge後のmainを使用する。

| 守る契約・期待値の根拠 | 検出したい故障 | 層・再利用 | 予算 |
|---|---|---|---|
| 固定されたメートルcourt、非対称3camera・非直線ballの独立投影から、既知の焦点距離/姿勢/位置摂動と正負のfractional offsetを復元 | camera交換、half-turn/時刻符号誤り、gauge未固定、ball残差の脱落 | 新unit `ball_calibration/test_bundle_adjustment.py`、24–36frame | solver代表2case、各5秒目標、CPUのみ |
| fitは元homography採用KPのみ、未採用KPはoptimizerから隔離する。x_reflectedは既存Hで元templateへ戻っている | holdout混入、二重反転、同名cameraの取り違え | 同unit、非対称KP subset/破壊したholdout | 3case、各1秒以下 |
| 時刻再標本化はt-offset、隣接実観測のみ、端/欠測を跨がない | sign反転、欠測補完、範囲外外挿 | 同unit、手計算する非対称2D座標 | 2case、各0.1秒以下 |
| court幾何不足・非有限設定・solver失敗は明示的に拒否 | 静かなbaseline代替、退化校正を採用 | 同unit、最小拒否fixture | 8case以下、各0.1秒以下 |
| 公開config→definition→assembler→BA→実codecを通し、camera/offset/maskを保存。LOVOは保持viewだけ | 設定無視、保存脱落、held-out evidence漏れ | 新integration、既存 `test_ball3d_bundle.py::test_real_component_codec_bundle_and_cpu_runner_roundtrip` を再利用 | 3case、各5秒目標 |
| fixed既定の位置・mask・inlier・理由・再投影配列が変更前mainと厳密一致 | 既定経路/設定/入出力schemaの意図しない変更 | integration、変更前mainの小さな固定fixtureの期待配列、既存baseline unit再利用 | 1case、1秒目標 |

新規は合計20件程度を上限とし、直積parametrizeを避ける。`--collect-only` と
`--durations=0 --durations-min=0`（`-n 0`、必要時も最大`-n 2`）で全node/件数/phase時間を保存する。
setupは小fixtureのみ、GPU/動画/実資産はunit・integrationへ持ち込まない。
時刻の符号を一時反転する故障注入で数値assert失敗を確認し、復元して再実行する。

実資産は別のCPU replayで4fold・3区間・LOVO coverage/common cohort、court holdout、
変更camera/offsetのtyped再読込、既存baselineの配列一致を確認する。
全method予測保存後にGTを採点側だけで読み、設定は事前固定する（後半を使った調整なし）。
動画は共有rendererで逐次処理する。比較図/動画/HTMLはoutputsの新directoryへ出し、
checksum・資源実測・目視検証と再現scriptをknowledgeへ記録する。

## 最終接続確認で追加した回帰

v1の保存内容を確認し、source-frameの2Dと時刻補正後の2Dを区別する契約を追加した。
新しいcaseを増やさず、公開config→sceneの既存integrationで、非ゼロoffsetでも
ball artifact/SceneResultの画素・maskが元観測と一致し、再標本化座標は診断欄にだけあることを確認する。
v2を新規directoryで実行し、v1の位置・valid・使用view・camera・offsetのbitwise一致も確認する。
v1は上書きせず、測定記録をv1/とoutputsへ保持する。設定や最適化の残差は変更しない。

## CIの設定監査による追加確認

runtime型の新fieldにPython defaultを置くと、構成所有の既定値という既存契約に反する。
型のdefaultを削除し、from_configでYAMLから決めた値を常に渡す。
既存の `tests/unit/utils/configuration/test_contracts.py::test_adapter_fields_have_truthful_default_precedence_and_absence_policy`
を再利用して契約を守り、新規caseは増やさない。収集はrepo標準のimportlibモードを明示し、
ミラー配置の同名test moduleを衝突させない。

CIの別数値環境では、記録したmainのfloat32位置と1 ULP程度の差が出ることも確認した。
許容差を広げず、checksum固定の元main計算核を同じNumPy/SciPyで動かし、全trajectory配列を
厳密比較する回帰へ変更する。元frameの画素・maskの固定期待値も引き続き厳密比較する。
凍結したgeometryとreconstructionはmainの原文で、現行の計算核を参照しない。
