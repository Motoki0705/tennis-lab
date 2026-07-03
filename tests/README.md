# `tests`

`tests/` は、このリポジトリの検証を `unit/`, `integration/`, `e2e/` の3軸で管理する入口です。AI がテストを追加するときは、まずここを見て「どこに置くか」を決めてください。

## 何を実装しているか

- `tests/unit/`: `src/` のうち、純粋関数や軽量な shape 変換のように入出力が明確な実装を検証する。
- `tests/integration/`: 複数モジュールの接続、最小データでの 1-step train/inference のような smoke test を置く。
- `tests/e2e/`: script や pipeline の entry point をコマンドレベルで検証する。
- `tests/conftest.py`: 全体で共有する fixture を置く。タスク固有 fixture は必要に応じて各サブツリーの `conftest.py` に分ける。

現在の実装もこの方針に沿っており、`unit` では `utils`, `tasks/base`, `tasks/ball_detection`, `tasks/court_detection`, `tasks/plcs` を、`integration` では `tasks/base` の smoke test を、`e2e` では `ball_detection` の entry point を扱っています。

## どこに追加するか

- `src/utils/geometry/angles.py` のような純粋ロジックは `tests/unit/utils/geometry/test_angles.py` に置く。
- `src/tasks/...` 配下でも、軽量で局所的なロジックなら `tests/unit/tasks/...` に mirror する。
- 学習 loop, 重い forward, rendering, 実運用に近い module 接続は `tests/integration/...` に置く。
- `scripts/` や pipeline 全体の検証は `tests/e2e/...` に置く。

mirror するのは `unit/` だけです。`src/` を 1 段落として同じ階層を保ち、ファイル名は `test_<module>.py` にします。

## mirror しないもの

- `configs/**/*.yaml`: Python の unit test と 1:1 対応させない。必要なら config compose の観点で `integration` に置く。
- `src/tasks/blcs/generate_dataset/webui/`: Next.js / TypeScript のため `tests/` には mirror しない。
- 空の `__init__.py`: test を作らない。

## 追加時の実務ルール

- すべての `src/` モジュールに機械的に `test_*.py` を作らない。
- 優先度は `utils/` → `tasks/base/` → 各 task の純粋ロジック。
- 置き場所に迷ったら「純粋関数なら `unit`、重い処理や接続確認なら `integration`、entry point なら `e2e`」で判断する。
