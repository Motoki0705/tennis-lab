[PR #2056](https://github.com/Motoki0705/tennis-lab/pull/2056) を作成し、レビュー可能にしました。`Closes #2031` 記載済みです。

外部I/Oのロック外移動、intent保存後のjob/run照合、schedule競合検出、clientの範囲・鮮度分離を実装しました。#2029の契約は維持しています。

- branch: `codex/dispatcher-triage-io`
- HEAD: `a3b490f2e7`
- 検証: 関連1984件、後続差分60件・19件が成功。追加回帰28ケース。pre-commit・ruff/mypy成功。
- [対象HEADのCI（Python）](https://github.com/Motoki0705/tennis-lab/actions/runs/43620117358): success
- 作業ツリーはclean。validatorは指定なし、試行0・完了0回。

[各pushの検証記録](https://github.com/Motoki0705/tennis-lab/issues/2031#issuecomment-6881541160) を残しました。残作業は親担当による統合レビュー・merge・merge後のmain CI確認です。