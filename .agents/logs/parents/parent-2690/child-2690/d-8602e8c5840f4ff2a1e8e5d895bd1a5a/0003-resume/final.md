正式レビューの major 3件を修正し、ready PR [#2717](https://github.com/Motoki0705/tennis-lab/pull/2717) に push しました。PR本文とリモートの HEAD は `52f5f06fbc89490f29c1578f220ea8a0d9b2dc06` で一致しています。

1. **phase 監査**。未知の outcome と call 欠落を `summarize`、`inspect_shards`、`build_profile`、`update-profile` の全入口で拒否し、既存 profile を保全します。正当な setup skip・xfail・失敗の分類も回帰テストで確認しました。
2. **候補 run の探索**。`--latest-main` は不正な plan や coverage を run ID・理由付きで記録し、次の候補を検査します。明示指定した run の不正、API 通信障害、ローカル I/O 障害ではエラー終了し、既存 profile を保全します。
3. **A/B/C の再比較**。計測元の26,189件、Actions の26,208件、修正後の実収集26,218件をそれぞれ固定して再計算しました。追加10件は `test_ci_shards.py` の6件と `test_ci_metrics.py` の4件です。いずれも計測済みのファイル単位 group に入り、各条件の割当・推定負荷・未計測件数は3時点で同じでした。 [3言語の文書と比較データ](https://github.com/Motoki0705/tennis-lab/blob/52f5f06fbc89490f29c1578f220ea8a0d9b2dc06/scripts/ci_metrics/sharding.ja.md#後続の全件inventoryに対する比較)

| 条件 | 最大推定 phase 秒 | 最大／平均 | 未計測 file／split case |
|---|---:|---:|---:|
| A | 610.580 | 1.000001 | 281／28 |
| B | 1,809.091 | 1.329619 | 0／0 |
| C | 1,360.609 | 1.000001 | 0／0 |

回帰テストを10件追加し、指定の6ファイルは通常環境で **390 passed（19.99秒）**、CI と同じ pytest 9.2.0 で **390 passed（21.38秒）**。Ruff、mypy、pre-commit も通過しました。固定 run からの profile 再生成はバイト単位で一致し、既存の比較6 run の監査も再計算して照合済みです。

新しい HEAD の CI は実行中です。親のレビュー進行に引き渡します。 [CI run 46067766284](https://github.com/Motoki0705/tennis-lab/actions/runs/46067766284)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}