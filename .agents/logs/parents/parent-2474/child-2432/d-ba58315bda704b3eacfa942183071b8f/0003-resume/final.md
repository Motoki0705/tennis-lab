親のタイムラインとラベル付き一覧を照合するよう修正しました。証拠用 Issue がない場合は `no_binding` で正常終了し、候補番号が一致しない場合だけ再試行します。#2477 を含む main を取り込み、同じブランチ `codex/parent-2474-issue-2432-primary` に push しました。 [PR #2478](https://github.com/Motoki0705/tennis-lab/pull/2478)

実 GitHub の dry-run では、親 #2390 の取得が29 GET から5 GET に減り、証拠のないクローズ済みの #2392 は3 GET で正常終了しました。統合後の関連テスト151件、pre-commit、actionlint が通過しています。

HEAD は `72d44d1d2710b5112a07721eff99e3aacda7d428` で、CI は実行中です。実 GitHub での自動クローズ確認と既存25件への `--apply` はマージ後に残っています。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2478,"head":"72d44d1d2710b5112a07721eff99e3aacda7d428","resume_on_success":false}}