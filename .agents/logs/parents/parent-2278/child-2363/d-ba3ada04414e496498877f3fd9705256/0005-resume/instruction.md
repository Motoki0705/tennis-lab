# 親からの指示: PR #2433 の CI 失敗の修正

PR #2433（HEAD 1f4929edf）の `Python` CI が失敗した（run 45101346095）。失敗は次の2件で、どちらも設定の監査（`src/utils/configuration`）に関するもの。

1. `tests/unit/utils/configuration/test_audit.py::test_repository_source_satisfies_current_audit`
   - `AuditReport.boundary_issues` に `src.tennis_scene.scripts.qualify_gvhmr_tenso...`（今回追加した script）の `BoundaryAuditIssue` がある（`NonHydraPathBoundary.validate` 関連）。
2. `tests/unit/utils/configuration/test_contracts.py::test_all_inventoried_runtime_boundaries_expose_truthful_authorities`
   - boundary の集合に、期待にない項目がある（Extra items in the left set）。

`Python` job の失敗は、上の2件の集約による。

## やること
- 今回追加・変更した実行入口（script・CLI）を、`src/utils/configuration` の README が定める実行境界の規約に従って正しく登録・実装する。
  - テスト側の期待値を書き換えて通すのではなく、規約に合わせる。
  - 規約上、その入口を常設すべきでないなら、knowledge 側に移すなどで整理する。どちらにしたかと理由を最終報告に書く。
- 失敗した2テストと、`tests/unit/utils/configuration` 全体をローカルで通す。
- push し、CI 待ちを宣言して終える（`resume_on_success: true`、next_step は「ドラフト解除と最終報告」）。

## やらないこと
- E2E の再実行はしない。14 clip の停止の扱いは、親が判断する。
- `gh run rerun` はしない。
