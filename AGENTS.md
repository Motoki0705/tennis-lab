# AGENTS.md: tennis-lab リポジトリ作業ルール (AI Agent / 人間共通)

## 0. TL;DR（必読）

*   **Hydra必須**（`scripts/` で `argparse` 禁止）
*   **main直作業禁止**（必ず適切なブランチをmainから切ること）
*   作業は必ず **main から新規ブランチを切って** 行う

### ブランチ命名規則
**形式：**
`<type>/<task>-<short-desc>`

* **type**: 変更の種類（例：`feature`, `fix`, `chore`, `docs`, `refactor`, `exp`）
* **task**: 対象領域（例：`wasb`, `plcs`, `blcs`）
* **short-desc**: 内容が分かる短い説明（英小文字 + `-` 推奨）
---

## 1. コンテキストと環境 (Context & Env)

### ディレクトリ構造
ソースコードは `src/` 配下にあり、タスクごとに分かれています。

*   `src/base`: 複数タスクで共有する抽象化・共通I/F（例: `BasePredictor` など）
*   `src/blcs`: BLCS（Ball Localization in Court System）
*   `src/plcs`: PLCS（Player Localization in Court System）
*   `src/wasb`: WASB/HRCNet 等の半自動アノテーション／データセット拡張
*   `src/utils`: 共有ユーティリティ（`geometry/`, `rendering/` など）

### 共通実装
複数タスクで共通するロジックは、必ず `src/` 直下または `src/base/`, `src/utils/` に実装し、docstringでその旨を明文化すること。**タスク内での再実装は禁止。**

### 依存・環境
*   **依存管理**: `pyproject.toml`（ロックファイルは `uv.lock`）
*   **Git操作**: `git`

---

## 2. サンドボックスと禁止事項 (Critical Constraints)

*   **Hydraの強制**
    *   `scripts/` 配下のスクリプトでは構成管理に必ず **hydra** を使用すること。`argparse` の使用は厳禁。

## 3. スクリプトとドキュメント規約（scripts/）

*   **エントリポイント**: 実行可能なスクリプトは `src/{task}/scripts/` に配置する。
*   **Docstring必須**: `scripts/` 配下のファイルには、必ずモジュールレベル docstring を記述する。含める内容：
    *   スクリプトの目的
    *   Hydraパラメータの説明