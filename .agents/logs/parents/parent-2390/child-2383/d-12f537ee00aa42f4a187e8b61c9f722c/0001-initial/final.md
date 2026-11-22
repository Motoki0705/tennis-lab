#2383 を `codex/parent-2390-issue-2383-primary` の `43381eadc` に実装して push しました。実装PRは現HEADの `Python` CI待ちです。 [PR #2395](https://github.com/Motoki0705/tennis-lab/pull/2395) · [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45024587434)

検証用ドラフト #2396 では、CPUマトリクス、システム依存、Lint、`Python` 集約、Web UIが省略されました。`Knowledge library` は42秒で成功し、構造・履歴等の検査は8秒、再現パス検査は2秒でした。実際の結果ファイルでHEADと比較元の一致を確認し、どちらかを変えると拒否されることも確認しました。証拠を #2395 に追記し、ドラフトを閉じました。 [ドラフト #2396](https://github.com/Motoki0705/tennis-lab/pull/2396) · [結果ファイル](https://github.com/Motoki0705/tennis-lab/actions/runs/45024770222/artifacts/14754787646)

ローカルでは pytest 671件、pre-commit、actionlint が成功しました。新規37件には、予約済み連番との衝突、不正な削除記録、再現参照切れ、範囲をまたぐrename、比較元不明の検証を含みます。同じ差分の `main` push 入力では通常CPU経路が選ばれました。実マージ後の `main` CI は #2390 の統合検証で確認します。

#2386 からは `PYTHONPATH=.agents/ops` で `lib.knowledge_ci` を利用できます。 [API仕様](https://github.com/Motoki0705/tennis-lab/blob/43381eadc7b6ac3d09b74249722a88d1ba270a0a/.agents/ops/lib/README.md#L26)

- `classify_pr(root: Path, *, head_sha: str, base_sha: str, merge_sha: str) -> Scope` は、全履歴と3つの完全SHAから、PR差分とテスト用マージ差分を調べ、判定と理由を返します。
- `artifact_name(run_id: int, run_attempt: int) -> str` が返す `knowledge-records-<run>-<attempt>` に `knowledge-records.json` が入ります。
- `verify_result(data: bytes | None, run: Mapping[str, object], *, pr_number: int, head_sha: str, base_sha: str) -> None` は、取得した結果とActions RESTのrunを照合し、拒否時は理由付きの `ValueError` を返します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2395,"head":"43381eadc7b6ac3d09b74249722a88d1ba270a0a","resume_on_success":false}}