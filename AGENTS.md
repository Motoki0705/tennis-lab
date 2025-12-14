AGENTS.md
tennis-lab リポジトリにおける作業ルール（AI Agent / 人間共通）
0. TL;DR（必読）

Hydra必須（scripts/ で argparse 禁止）

GPU/大規模DL禁止（ML_SANDBOX_DISABLE_GPU 想定、テストは tests/fixtures/ を使う）

main直作業禁止（必ずブランチを切る）

チェックボックス駆動：1項目ごとに「実装 → テスト設計/作成/実行 → pre-commit修正 → commit → [x]」

agents_workspace/work/ は tmp（非コミット）。PR作成後に削除する



---

1. コンテキストと環境 (Context & Env)

ディレクトリ構造: ソースコードは src/ 配下にあり、タスクごとに分かれている。

src/base: 複数タスクで共有する抽象化・共通I/F（例: BasePredictor など）

src/blcs: BLCS（Ball Localization in Court System）

src/plcs: PLCS（Player Localization in Court System）

src/wasb: WASB/HRCNet 等の半自動アノテーション／データセット拡張

src/utils: 共有ユーティリティ（geometry/, rendering/ など）


共通実装: 複数タスクで共通するロジックは、必ず src/ 直下または src/base/ に実装し、docstringでその旨を明文化する。タスク内での再実装は禁止。

依存: pyproject.toml（ロックは uv.lock）で管理。

コマンド:

Python系は uv run

git操作は git

PR作成は gh




---

2. サンドボックスと禁止事項 (Critical Constraints)

Hydraの強制: scripts/ 配下のスクリプトでは構成管理に必ず hydra を使用すること。argparse の使用は厳禁。

学習の制限:

ML_SANDBOX_DISABLE_GPU 環境下を想定。GPUを使用した重い学習や推論は禁止。

大規模なデータセットのダウンロードや生成は禁止。テストは tests/fixtures/ のダミーデータを使用する。


既存テストの保護: tests/integration/ はパイプラインの健全性を保証するため、原則として破壊・削除しない（ロジック変更に伴う最小修正のみ許容）。



---

3. ワークフロー (Workflow)

tennis-lab の開発は、以下のサイクルを標準とする。
AI Agent は各フェーズの内容を agents_workspace/work/ に記録し、チェックボックス駆動で進めること。

3.1 ブランチ作成

1. main に移動し最新化する


2. 新規ブランチを作成する（feature/, fix/, refactor/, exp/ 等）



main で直接作業してはならない。


3.2 設計（設計書 + 実装計画 + テスト計画）

実装前に以下を明確化すること：

目的（What / Why）

変更・追加するモジュール（src/base / src/{task} 等）

共通化判断（再実装禁止の観点で、どこに置くべきか）

影響範囲（unit / integration）

制約チェック（Hydra / GPU禁止 / integration保護）


このフェーズで必ず以下を作成する（テンプレからコピーして埋める）：

agents_workspace/work/<branch_name>/design.md

agents_workspace/work/<branch_name>/implementation.md（チェックボックス形式の実装手順/TODO）

agents_workspace/work/<branch_name>/tests.md（責務・想定機能・テスト方針）


設計が固まったら 設計確定コミットを作成してよい。

3.3 実装（チェックボックス駆動）

実装は implementation.md のチェックボックスを最小単位として進める。
チェックボックス1項目 = 1作業単位であり、以下を必ず含む：

1. 実装（コード変更）


2. tests.md に テスト設計を追記（責務/想定機能/ケース）


3. テスト作成（原則 unit、必要なら integration）


4. テスト実行


5. tests.md に 実行結果を記録（コマンド / PASS・FAIL / 要点）


6. git commit（ここで pre-commit が走る）


7. pre-commit が失敗した場合は修正し、再度 commit する（通るまで繰り返す）


8. commit が成功したら、そのチェックボックスを - [x] に更新する



> チェックを付ける条件：テスト実行が完了し、pre-commitを通過したコミットが作成されていること。



3.4 PR下書き → PR作成

作業完了後、agents_workspace/work/<branch_name>/pr.md を埋める（テンプレから作成）

gh でPRを作成する

PRには以下を明確に記載する：

変更の目的

設計上の判断

テスト範囲（unit / integration）



3.5 PR後の後片付け（必須）

agents_workspace/work/ は tmp扱いであり、リポジトリに残さない。
PR作成後に必ず以下を実施する：

agents_workspace/work/<branch_name>/ を削除する

agents_workspace/work/ は コミットしない（.gitignore 対象）



---

4. agents_workspace 運用（tmp方針）

agents_workspace/templates/：テンプレ置き場（コミット対象）

agents_workspace/work/：作業ログ（非コミット / tmp、PR後削除）


4.1 作成ルール

agents_workspace/templates/ のテンプレをコピーして work/<branch>/ を作ること。

work/<branch>/ は少なくとも以下を含むこと：

design.md

implementation.md

tests.md

pr.md




---

5. スクリプトとドキュメント規約（scripts/）

エントリポイント: 実行可能なスクリプトは src/{task}/scripts/ に配置する。

Docstring必須: scripts/ 配下のファイルには、必ずモジュールレベル docstring を記述する。含める内容：

スクリプトの目的

実行コマンド例（uv run ...）

Hydraパラメータの説明


Hydra必須: scripts/ で argparse を使用しない。



---

6. テスト方針（運用ルール）

変更ロジックには原則 unit test を追加する（tests/unit/）。

データやモデルは tests/fixtures/ の再利用を優先する。

パイプラインに影響する場合のみ integration test を実行する（tests/integration/）。

tests/integration/ の既存テストは原則として破壊・削除しない。


テストに関しては agents_workspace/work/<branch>/tests.md に以下を必ず残す：

対象モジュールの責務・想定機能

テスト設計（ケース、境界、例外）

実行コマンド

実行結果（PASS/FAIL と要点）

integration を実行した/しない判断（理由）



---

7. pre-commit（品質ゲート）

git commit により pre-commit が実行され、lint/format/typeチェックが走る。

pre-commit が失敗した場合、Agent は 修正して再コミットし、成功するまで先へ進まない。

pre-commitが通ったコミットが作成されて初めて、対応するチェックボックスを完了扱いとする。