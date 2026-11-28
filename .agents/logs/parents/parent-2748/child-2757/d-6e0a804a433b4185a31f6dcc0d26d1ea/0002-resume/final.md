正式レビューの major 6件を修正し、同じブランチに push して ready PR の説明を更新しました。 [PR #2762](https://github.com/Motoki0705/tennis-lab/pull/2762)

ブランチ `codex/parent-2748-issue-2757-primary`  
HEAD `27672d5d25adf612eb10b4203dea1fc7d4a659c2`

1. 幾何・ネット通過の参照を同一クリップに、サーブ試行の打球参照を同一ポイントに制約しました。
2. V4 の採否と `in_point` を分離し、除外理由を必須にしました。V4 とスコアから導出できる終了原因は `net_error` と `double_fault` に限定しました。
3. 観測着地・推定着地・欠測の整合性を DB で検証し、ネット通過に使用したプロファイルを試合エクスポートに含めました。
4. `(match_context_id, match_point_key)` で、重複クリップが捉えた同一ポイントを一意化しました。
5. 条件別ミス率に `shot_role` を必須化し、同じ高さ帯でもサーブと非サーブの分母が分かれることを確認しました。
6. 指標ごとの計数関係と根拠 ID を、保存ヘルパー・ORM・CLI で検証します。直接 SQL で注入した重複根拠も JSON／CSV 出力前に拒否しました。

未マージの初期リビジョン `0001` を更新しました。Compose の PostgreSQL 17.12 で `alembic upgrade head → downgrade base → upgrade head` と `alembic check` が成功しました。`pytest -n 0 -m postgres tests/integration/tennis_scene/analytics` は8件（4.62秒）、`spin test --serial -q --no-cov` の対象 CPU テストは113件（1.65秒）、CI 構成・集約テストは23件（9.90秒）が合格しました。pre-commit の Ruff と mypy も通過しました。

検証用 Compose は停止し、ボリュームを保持しました。新 HEAD の CI は実行中です。 [CI 実行](https://github.com/Motoki0705/tennis-lab/actions/runs/46167762956)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}