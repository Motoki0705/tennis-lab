正式レビューの指摘を修正し、`codex/parent-2390-issue-2383-primary` にコミット `6d60b29184cd626d47957c4448f170d1e2b81127` を push しました。READMEとPR本文を更新し、新HEADのCIが起動したことを確認しました。 [PR #2395](https://github.com/Motoki0705/tennis-lab/pull/2395) · [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45027808325)

更新後のAPIは `verify_result(data: bytes | None, run: Mapping[str, object], *, pr_number: int, head_sha: str, base_sha: str, merge_sha: str) -> None` です。`classify_pr` に渡したテスト用マージSHAと結果ファイルの `merge_sha` に完全一致を求め、不一致は理由付きの `ValueError` で拒否します。

有効な40桁SHAへの差し替えを拒否する回帰テストを1件追加し、対象のunit・E2E 38件とpre-commitのruff・mypyが通りました。#2396 の実ファイルでも、一致時の受理と、結果側または照合入力側のマージSHAだけを変えた場合の拒否を確認しました。 [結果ファイル](https://github.com/Motoki0705/tennis-lab/actions/runs/45024770222/artifacts/14754787646)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2395,"head":"6d60b29184cd626d47957c4448f170d1e2b81127","resume_on_success":false}}