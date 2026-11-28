#2757 を実装し、`Closes #2757` を含む ready PR を作成しました。PostgreSQL 17 の18表、Alembic 初期マイグレーション、版付きの31指標、DB 照会と JSON／CSV エクスポートを追加しました。複合外部キーでチームをまたぐ参照を防ぎ、V4 の除外イベントでは元 ID と根拠を保持し、欠けている項目を NULL で保存します。 [PR #2762](https://github.com/Motoki0705/tennis-lab/pull/2762)

- ブランチ `codex/parent-2748-issue-2757-primary`
- HEAD `a175d094f6a7ff0e478601d323244a76dd64a6aa`
- Compose の PostgreSQL 17.12 で `alembic upgrade head → downgrade base → upgrade head` と `alembic check` が成功。PostgreSQL 結合テスト5件（4.91秒）で全18表の CRUD、制約違反の拒否、CLI 照会と JSON／CSV の一致を確認しました。
- CPU テスト99件（2.13秒）、CI 構成・集約テスト23件（10.44秒）、pre-commit の Ruff・mypy が通過しました。`git check-ignore` で追加ファイルも照合しました。

検証用 Compose は停止し、名前付きボリュームは保持しました。PR の CI は起動済みで、完了結果は未確認です。 [CI 実行](https://github.com/Motoki0705/tennis-lab/actions/runs/46165574579)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}