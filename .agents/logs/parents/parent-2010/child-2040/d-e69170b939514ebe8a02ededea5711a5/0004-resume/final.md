[PR #2054](https://github.com/Motoki0705/tennis-lab/pull/2054)で、両指摘を成立と確認し修正しました。

- **pr-2054-1**: token取得をregistryロック外へ移動。復帰時の競合検出と、競合後のstale保存拒否を維持しました。
- **pr-2054-2**: insteadOf・pushInsteadOf適用後の全fetch/push URLを照合し、別repo・非GitHub・取得失敗を明示エラーにしました。

回帰テスト12ケースを追加。`pytest -n 4`で関連150件成功（97.55秒）、明示ruff・mypy・`pre_commit run --files`も成功しました。[実行コマンド全文と検証結果](https://github.com/Motoki0705/tennis-lab/pull/2054#issuecomment-6888096390)

branch: `codex/gh-http-client`  
新HEAD: `3124e10c6d0d1e46674fb44b9d1529a62d851f7b`（push済み、clean）

[対象HEADのPython CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43669006698/job/132679436545)は成功。CPU全10 shard・Lint・System dependency testsも成功し、PR本文を更新済みです。

残作業は親による統合レビュー・merge・merge後mainのCI確認です。追加validatorは未指定、試行／完了0／0です。