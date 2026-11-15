[PR #2149](https://github.com/Motoki0705/tennis-lab/pull/2149) を作成・push済みです。

- branch: `codex/review-resolve-base-merge-scope`
- commit: `ee23fdad9`
- resolve・inspect・gateの確認範囲を共通化。公開記録を件数・digestに縮約し、全パスをローカル保存します。旧記録は旧定義で検証します。
- ローカル730件・pre-commit成功。[CI #4277](https://github.com/Motoki0705/tennis-lab/actions/runs/43724946564) も`Python`を含む全14 job成功。作業treeはcleanです。
- 追加のops単独mypyは既存12診断、増分0です。

固定実例は1,532パスから確認対象82パスになりました。[範囲JSON]([local path])

残りは親の統合レビュー・mergeと、#2085の最新範囲を確認した実resolve/gate、merge後main CI・運用再開の確認です。実PRへのresolve/gate、systemd・git-sync操作は行っていません。