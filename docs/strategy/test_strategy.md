# Test Strategy

* **目的**: 変更をテストで裏付け、回帰を防ぐ。
* **階層/配置**:

  * `tests/unit/` … **役割別に細分化**（例: `tests/unit/models/`, `tests/unit/utils/`, `tests/unit/io/`）。
  * `tests/integration/`
  * `tests/e2e/`（任意・遅い）
* **マーカー例**: `unit`, `integration`, `e2e`, `slow`。
* **再現性（必須）**: 乱数seed固定／外部I/Oはfixtureでスタブ／時間・環境依存はモック。
* **実行ポリシー**:

  * **ローカル/PR**: `unit` 必須。PR作成時に**integration要否を評価し、必要なら実行**。
  * **夜間/手動**: `e2e` と `slow`。
* **失敗時フロー**: 原因仮説→修正→**表層回避**→テスト強化。
* **品質ゲート**: Coverage **≥ 75%**（段階的引き上げを検討）。
