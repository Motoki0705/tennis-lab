# Git Workflow

* **方針**: Trunk-Based。作業開始時に適切なブランチにいるか確認→不適なら切替 or `main`から新規作成。
* **ブランチ命名**: `<type>/<scope>-<short>`（例: `feat/model-loader`, `fix/utils-npe`）。*チケットIDは任意*。
* **マージ方式**: **Merge commit** を標準。Rebase/Squashは使用しない方針。
* **コミット単位**: 原則「**1ファイル=1コミット**」。ただし**integrate過程の赤→緑化**やテスト修正では**例外を許容**（連続修正OK）。
* **コミット規約**: Conventional Commits。ヘッダのみ。

  ```
  <type>(<scope>): <何をしたか1行で>
  ```
* **pre-commit**: `ruff`/`mypy`を必須。未通過のコミット/PRは禁止。
* **PR**: 小さく・早く。**自己レビュー**で確認→PR作成→Merge。PR作成時に**integrationテストの要否を評価**。
* **CI（最小ゲート）**: Lint/Type/Unit必須。Coverage **≥ 75%** でFail。
