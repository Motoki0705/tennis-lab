ready PR [#2717](https://github.com/Motoki0705/tennis-lab/pull/2717) を作成しました。`Closes #2690` を含む日本語本文と、指定ブランチ `codex/parent-2690-issue-2690-primary` の HEAD `3060eaa20705e5ad0dfa6aca8e4ea609fc4ffc0d` を確認済みです。

| 比較 | 最長 Test ステップ | 必須 `Python` 完了 | 全 runner 時間 |
|---|---:|---:|---:|
| 全件 A/C 各2回の平均 | 1,611 → 962秒（**40.3%短縮**） | 1,774 → 1,114.5秒 | 8,792 → 9,134秒（**3.9%増**） |
| Scoped PR A/C 各1回 | 1,069 → 706秒（**34.0%短縮**） | 1,208 → 823秒 | 6,496 → 6,328秒（**2.6%減**） |

全件の各組で最長 Test ステップが **46.8%・32.4%** 短縮し、両方とも30%目標を達成しました。全66テスト job で uv cache hit を確認しました。ホストの CPU 型番と Ubuntu image の版にはばらつきがあり、その条件も記録しています。 [3言語の文書と比較データ](https://github.com/Motoki0705/tennis-lab/blob/3060eaa20705e5ad0dfa6aca8e4ea609fc4ffc0d/scripts/ci_metrics/sharding.ja.md#同一条件のactions比較)

6 run の worker event と JUnit から完全性監査を再計算しました。各 scope で要求・実行 nodeid が一致し、欠落・重複・余分な実行は0件でした。skip 対象も scope 内で一致し、`system_deps` は毎回17件すべて passed でした。 [全 shard の計測・監査データ](https://github.com/Motoki0705/tennis-lab/blob/3060eaa20705e5ad0dfa6aca8e4ea609fc4ffc0d/scripts/ci_metrics/sharding-comparison.json)

最終HEADのローカル検証は **380 passed（24.02秒）**。Ruff、mypy、実装ファイルを指定した pre-commit、3言語の数値・再現コマンドの照合も通過しました。最終PRの CI と正式レビューは親のレビュー進行に引き渡します。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}