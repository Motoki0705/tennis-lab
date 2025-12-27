# AGENTS.md: tennis-lab リポジトリ作業ルール (AI Agent / 人間共通)

## 0. ブランチング戦略とワークフロー
**main直作業禁止**（必ず適切なブランチをmainから切ること）。

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
*   `src/blcs`: 画面上のシーケンシャルなボールの位置とコートのキーポイントを用いて3d上のボールの軌道を推定するモデルの学習環境やそのデータセットをシミュレーションによって生成する。
*   `src/plcs`: 画面上のプレーヤーのキーポイントとコートのキーポイントを用いて3d上のプレーヤーの位置と回転を推定するモデルの学習環境やそのデータセットをシミュレーションによって生成する。
*   `src/wasb`: ボールの位置を画面上の点として特定するモデルの学習環境などを提供する
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
*   **Docstring必須**: `scripts/` 配下のファイルには、必ずモジュールレベル docstring を記述する。例えば以下のようなフォーマットで書くこと。
```python
"""Generate PLCS scenes for training using Hydra-managed configuration.

Example commands:
    `uv run python -m src.plcs.scripts.generate_dataset`
    `uv run python -m src.plcs.scripts.generate_dataset run.output_dir=data/plcs simulation.num_scenes=10`

Config entry point: `src/plcs/configs/generate_dataset.yaml`
"""
```

## 4. 使用可能なツール

### `agents_workspace/sub_agents/pre_commit_subagent.sh`
変更ファイル（`git diff --name-only HEAD`）に対して `pre-commit` を実行し、失敗した場合は Codex Sub-agent に修正を委譲します（通常は `uv run --no-sync pre-commit ...` を実行）。ログは `agents_workspace/sub_agents/logs/`（または `CODEX_SUBAGENT_LOG_DIR`）に保存され、標準出力には 1 行の JSON を返します。

**実行例**
```bash
bash agents_workspace/sub_agents/pre_commit_subagent.sh
```
（ヘルプ：`bash agents_workspace/sub_agents/pre_commit_subagent.sh -h`）

**想定出力（例）**
```json
{"status":"pass","fixed":false,"files_touched":[],"remaining_errors":[],"summary":"pre-commit passed","needs_main":false,"message_for_main":""}
```

### `agents_workspace/sub_agents/test_subagent.sh`（推奨：テスト用）
指定したテストコマンド（デフォルトは `uv run --no-sync pytest -q`）を実行し、失敗した場合はログから対象ファイルを抽出して Codex Sub-agent に修正を委譲します。ログは `agents_workspace/sub_agents/logs/pytest_*.log`（または `CODEX_SUBAGENT_LOG_DIR`）に保存され、標準出力には 1 行の JSON を返します。

**基本（デフォルトの pytest）**
```bash
bash agents_workspace/sub_agents/test_subagent.sh
```
（ヘルプ：`bash agents_workspace/sub_agents/test_subagent.sh -h`）

**テスト対象を絞る（例：単体テスト/ケース指定）**
```bash
bash agents_workspace/sub_agents/test_subagent.sh --test-cmd 'uv run --no-sync pytest -q tests/test_example.py::test_case'
```

**想定出力（例）**
```json
{"status":"pass","fixed":false,"files_touched":[],"remaining_failures":[],"summary":"tests passed","needs_main":false,"message_for_main":""}
```
```json
{"status":"fail","fixed":false,"files_touched":[],"remaining_failures":["..."],"summary":"...","needs_main":false,"message_for_main":""}
```
