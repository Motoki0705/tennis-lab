PR [#1758](https://github.com/Motoki0705/tennis-lab/pull/1758) を作成しました。未mergeです。

- branch: `codex/v3-fixed-geometry-triangulation`
- commit: `0ab7ea33c`
- 保存v3・固定幾何から、観測と推定を区別した三角測量を実装。旧モデル経路の契約を維持しました。
- 関連228テスト、pre-commit、[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42498140017)のLint・CPU全10分割・Pythonが成功。作業treeはcleanです。

指定bundleには `court_detection="verified_calibration_frame0"` が必要です。手順・証拠はPRと子issueへ記録済み。CLI接続 #1713、実データ受入 #1715 は未実施です。統合レビュー・merge・merge後main CI確認を親へ引き継ぎます。