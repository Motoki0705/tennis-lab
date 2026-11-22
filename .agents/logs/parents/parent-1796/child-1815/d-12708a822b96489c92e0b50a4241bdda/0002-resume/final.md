[PR #1871](https://github.com/Motoki0705/tennis-lab/pull/1871)を同branchで更新しました。HEAD: `ceaee2981`。

- **pr-1871-1**: 私的IDを捕捉前に登録し、公開要約を安全な項目へ限定しました。
- **pr-1871-2**: 入力不正と捕捉障害を分離し、障害時も元操作を維持して永続診断を残します。

両指摘の修正前失敗→修正後成功を確認済みです。追加12ケース、関連回帰384件・最終検証43件、pre-commit成功。[対象HEADのCI Pythonも初回で成功](https://github.com/Motoki0705/tennis-lab/actions/runs/42856454789)しました。

親による修正確認・指摘解消・mergeと、merge後の確認が残ります。