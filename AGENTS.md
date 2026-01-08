# AGENTS.md: tennis-lab リポジトリ作業ルール (AI Agent / 人間共通)

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

## 2. スクリプトとドキュメント規約（scripts/）

*   **Hydraの強制**: `scripts/` 配下のスクリプトでは構成管理に必ず **hydra** を使用すること。`argparse` の使用は厳禁。
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

## 3. 使用可能なツール

Sub-agent ツールの詳細な使い方はスキル化済みのドキュメントを参照してください。

- `skills/agents-consult/SKILL.md`
- `skills/agents-review/SKILL.md`
- `skills/agents-pre-commit/SKILL.md`
- `skills/agents-test/SKILL.md`

## 4. 推奨ワークフロー（ブランチ作成→相談→作業→レビュー→検査→テスト）

### AI Agent向け（必須）チェックリスト
このリポジトリでは、AI Agent はユーザーから明示されていなくても **必ず** 以下を満たしてから作業してください（逸脱する場合は、作業前にユーザーへ確認し、最終報告にも理由を明記すること）。

1) **ブランチ作成（必須）**
- **`main` / `master` / `develop` での直接編集は禁止**
- 変更を 1 行でも加える前に、現在ブランチを確認し、`main` 等であれば新規ブランチへ移動する

2) **開発前の相談（推奨）**
- 複雑なタスクの場合は、`src.agents.scripts.consult` を使用して複数のLLMにアプローチを提案させる
- 実行例は `skills/agents-consult/SKILL.md` を参照
- 複数の視点からのフィードバックを得ることで、より良い実装方針を決定できる

3) **検査とテスト（必須）**
- 変更後は必ず `src.agents.scripts.pre_commit` → `src.agents.scripts.test` の順で実行する（ユーザーが「実行しないで」と言った場合を除く）
- **テストは全て実行するのではなく、変更に影響するテストのみを `task.test_cmd` で指定して実行する**
  - 実行例は `skills/agents-test/SKILL.md` を参照
  - 変更したモジュールに対応するテストファイルを特定し、必要最小限のテストを実行する

4) **開発後のレビュー（推奨）**
- 変更完了後、`src.agents.scripts.review` を使用してコード変更をレビューさせる
- 実行例は `skills/agents-review/SKILL.md` を参照
- 問題点の発見、改善提案、新しいタスクの特定に役立つ

5) **ドキュメントの整合性確認（必須）**
- 変更内容が各タスクの `README.md` やドキュメントに影響する場合は、必ず更新する
- 特に `src/{task}/README.md` は実装と矛盾がないよう常に最新に保つ
- 新機能追加・API変更・設定変更があった場合は、関連ドキュメントの更新を確認する

6) **例外の扱い（必須）**
- 環境都合（例: `uv run` の権限エラー等）で推奨コマンドが失敗した場合も、回避策を適用して **同等の検査/テストを実行する**
- どうしても実行できない場合は、その理由と、代替で何を確認したかを最終報告に明記する

### ステップバイステップの手順

1) **ブランチを切る（main から）**

（`main` 直作業禁止）
```bash
git checkout main
git pull
git checkout -b feature/<task>-<short-desc>
```
命名規則（推奨）: `<type>/<task>-<short-desc>`（例：`feature/wasb-add-foo`）

2) **作業する**
- 変更・追加を行う（規約は本ドキュメントに従う）

3) **pre-commit を実行（ツール推奨）**
```bash
uv run python -m src.agents.scripts.pre_commit
```

4) **テストを実行（ツール推奨）**
- **変更に影響するテストのみを `task.test_cmd` で指定して実行する（全テスト実行は禁止）**
```bash
# 変更に関連するテストファイルを特定して実行（-n auto で並列実行）
uv run python -m src.agents.scripts.test 'task.test_cmd=uv run --no-sync pytest -q -n auto tests/unit/test_affected.py'

# 複数のテストファイルを指定する場合
uv run python -m src.agents.scripts.test 'task.test_cmd=uv run --no-sync pytest -q -n auto tests/unit/test_foo.py tests/integration/test_bar.py'

# 特定のテストケースを指定する場合
uv run python -m src.agents.scripts.test 'task.test_cmd=uv run --no-sync pytest -q tests/test_example.py::test_case'
```

5) **ドキュメントの更新確認**
- 変更がREADMEやドキュメントに影響する場合は更新する
- 特に `src/{task}/README.md` は実装と矛盾がないよう最新に保つ

---
## 5. `uv run` の Permission denied 回避（重要）

Codex 実行環境では、`uv` のデフォルトキャッシュが `/root/.cache/uv` を指し、権限の都合で `Permission denied` になることがあります。

### 推奨: `--cache-dir` を workspace 配下に固定
```bash
uv --cache-dir agents_workspace/tmp_cache/uv_cache run --no-sync pytest -q
uv --cache-dir agents_workspace/tmp_cache/uv_cache run --no-sync pre-commit run -a
```

### 代替: 一時キャッシュ（遅いが確実）
```bash
uv --no-cache run --no-sync pytest -q
```

### subagent 経由で回避する例
```bash
uv run python -m src.agents.scripts.test 'task.test_cmd=uv --cache-dir agents_workspace/tmp_cache/uv_cache run --no-sync pytest -q tests/...'
uv run python -m src.agents.scripts.pre_commit
```
