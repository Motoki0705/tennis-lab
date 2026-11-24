# 親からの続き（2026-11-24、担当親 Joseph）

前ターンの結果（原因分析の再現、ダブルス幅・滞在25%で 3D 選手frame +13,218、停止 14→11、Meiji 3cam dev 4 clip で変化0）を確認した。#2442 の全体実行が完了し、69 clip・1,205 frame・34,664 box の疑似正解を公開した。
- dataset: `[local path]`
- PR #2448 の新HEAD: `c1c9b06ab3ebf36b595066bc34377e85c561fd8b`

## このターンで行うこと
1. **CI の失敗を先に直す**。PR #2450（HEAD 068076542）の CPU tests で `tests/integration/tennis_scene/test_person_feature_artifacts.py::test_saved_features_exactly_match_pre_split_tracking_selection_and_gsi` が失敗した（rule の辞書が期待値と不一致。 https://github.com/Motoki0705/tennis-lab/actions/runs/45244426359/job/137937692501 ）。既定の規則の出力が変わっていないかを最優先で確かめる。opt-in の設定を追加したことで既定経路の保存内容が変わったなら、既定は変えない方向で直す。テストの期待値を書き換える場合は、既定の挙動が変わっていない根拠を示す。
   - 親が #2448 の新HEAD `c1c9b06a` をこのブランチへ merge した（ローカルの merge commit `c0fd294e`、未push）。そのまま使い、push する。
2. 試作した各候補（既定、ダブルス幅・25%、滞在率の変更、ほか試作したもの）を、疑似正解で採点する。
   - 選手の選別の precision・recall（box単位と選手frame単位）、誤選別の内訳（コート脇の人・隣のコート・審判など、分かる範囲）。
   - 層（疑義5層・確信層の無作為標本）別に報告する。人による抜き取りの誤り率は未記録なので「暫定」と明記する。
3. 候補の推奨と根拠をまとめ、既存経路と新経路（#2445 の3D軌跡による選別）の分担の案を書く。既定の規則は切り替えない（人間の判断事項）。
4. PR #2450 を更新する（本文は日本語、`Closes #2440` を維持）。PR番号と完全な HEAD を宣言して終了する。
