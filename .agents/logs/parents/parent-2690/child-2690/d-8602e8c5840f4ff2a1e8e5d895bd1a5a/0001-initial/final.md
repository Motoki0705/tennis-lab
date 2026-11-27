#2690 の Turn 1 を完了しました。新しい CPU profile、更新 CLI、growth report、回帰テスト、3 言語の文書を `codex/parent-2690-issue-2690-primary` に push しました。ローカルとリモートの HEAD は `3e9e0b7380b050f205f20372dd31f298dbe4a483` で一致し、worktree は clean です。 [コミット](https://github.com/Motoki0705/tennis-lab/commit/3e9e0b7380b050f205f20372dd31f298dbe4a483)

### 計測元と再生成

着手時点で最新の適格な main push は run `45955929416`、HEAD `8d629a35289f475e514291167435c333876afdd2` でした。集約・plan・全 11 shard の attempt はすべて `1` です。全 shard で 2 workers の観測と JUnit を再監査し、`success`・`full_cpu_success`・`observations_complete` がすべて `true`、`event=push`、`ref=refs/heads/main`、`deferred_count=0` と照合しました。26,189 件のうち 26,070 件が passed、119 件が既存条件による skip、失敗は 0 件です。専用 `system_deps` の 17 件はすべて passed でした。 [Actions run](https://github.com/Motoki0705/tennis-lab/actions/runs/45955929416)

- Lane plan artifact `15180220319`、coverage artifact `15181198448`
- 計測 artifact ID（shard 0〜10 順） `15181005137, 15180522011, 15180860195, 15180532163, 15181140369, 15180766930, 15180302726, 15181060041, 15181810195, 15180153128, 15179354987`
- 新 profile は 1,160 file、6,539 split case。file 重みは 282 件追加、878 件更新、観測 case が 0 件の旧エントリ 5 件を除外しました。 [profile と出典](https://github.com/Motoki0705/tennis-lab/blob/3e9e0b7380b050f205f20372dd31f298dbe4a483/scripts/ci_metrics/cpu-file-times.json)

実際に実行した更新コマンドです。`--latest-main` でも同じ run が選ばれ、生成ファイルの SHA-256 が一致しました。

```bash
.venv/bin/python -m scripts.ci_metrics update-profile --run-id 45955929416 --attempt 1 \
  --baseline [local path] \
  --output scripts/ci_metrics/cpu-file-times.json \
  --report [local path] \
  --split-file tests/e2e/agents_ops/test_board_messages.py \
  --split-file tests/e2e/agents_ops/test_parent_session.py \
  --split-file tests/integration/agents_ops/test_storage_tier_crash_boundaries.py \
  --split-file tests/integration/agents_ops/test_storage_tier_recovery_boundaries.py
```

### A/B/C の比較

通常 10 shard の setup・call・teardown の累積秒です。A 実測は計測元 run の割当、B/C は同じ inventory と観測値を使った再配分です。

| Shard | A 旧重み・推定 | A 実測 | B 重み更新・推定 | C case 分割・推定 |
|---|---:|---:|---:|---:|
| 0 | 610.579 | 725.360 | 1,809.091 | 1,360.607 |
| 1 | 610.580 | 1,322.624 | 1,310.776 | 1,360.609 |
| 2 | 610.579 | 611.440 | 1,310.776 | 1,360.608 |
| 3 | 610.579 | 1,304.961 | 1,310.776 | 1,360.609 |
| 4 | 610.579 | 1,185.038 | 1,310.777 | 1,360.608 |
| 5 | 610.578 | 1,397.866 | 1,310.777 | 1,360.607 |
| 6 | 610.579 | 1,650.712 | 1,310.776 | 1,360.608 |
| 7 | 610.580 | 844.635 | 1,310.778 | 1,360.609 |
| 8 | 610.578 | 3,311.545 | 1,310.777 | 1,360.608 |
| 9 | 610.578 | 1,235.024 | 1,310.778 | 1,360.609 |
| 最小 | 610.578 | 611.440 | 1,310.776 | 1,360.607 |
| 最大 | 610.580 | 3,311.545 | 1,809.091 | 1,360.609 |
| 平均 | 610.579 | 1,358.921 | 1,360.608 | 1,360.608 |
| 最大 / 平均 | 1.000001 | 2.436894 | 1.329619 | 1.000001 |

未計測の通常 file／split case は A が **281／28**、B と C が **0／0** です。実測 case コストを C に載せ替えたときの最大は **1,360.609 秒**で、A 実測の最大より **58.9%** 小さくなりました。 [全割当と集計 JSON]([local path])

| 重い file | ケース | 累積 phase 秒 | A → B → C の shard |
|---|---:|---:|---|
| `test_storage_tier_recovery_boundaries.py` | 4,606 | 1,809.091 | 8 → 0 → 全 10 shard |
| `test_storage_tier_crash_boundaries.py` | 1,649 | 533.836 | 6 → 1 → 全 10 shard |
| `test_parent_development.py` | 66 | 423.166 | 1 → 2 → 0 |
| `test_parent_completion_resume.py` | 21 | 373.922 | 8 → 3 → 1 |
| `test_court_dataset.py` | 29 | 355.775 | 0 → 4 → 2 |

**C は 4 file を分割します。** 旧宣言の `test_merge_gate.py` は補助モジュールで、収集 nodeid が 0 件でした。B では従来の分割方針が作用する 2 file を使い、C に停止・復旧の 2 file を追加しました。両 file は function scope の `tmp_path`・`monkeypatch`・seed fixture を使い、repo・mirror・模擬 `[local path]`・journal・lock は case ごとの root 配下です。trace hook は fork した子内で動作し、子を回収した後に pytest が `sys.path` を復元します。実際の収集 guard で **6,255 件すべてが一度ずつ割り当てられ**、未計測 case は 0 件でした。代表の停止・復旧 case も 2 workers で 2 件とも通過しました。 [検証と再現手順](https://github.com/Motoki0705/tennis-lab/blob/3e9e0b7380b050f205f20372dd31f298dbe4a483/scripts/ci_metrics/sharding.ja.md)

### ローカル検証と残件

更新 CLI は不適格な証拠を拒否し、検証とレポート生成を終えてから profile を原子的に置換します。growth report には profile 出典、未計測率、通常 shard の推定・実測負荷と最大／平均比を追加しました。回帰テスト 19 件を追加し、部分再実行、証拠不一致や書き込み失敗時のファイル保全、未計測 case の実行、full・scoped の完全性監査を検証しました。

- 指定の pytest 6 ファイルは **C で 380 passed（25.12 秒）**、旧 profile だけに差し替えた **A で 380 passed（25.66 秒）**。
- `ruff check scripts/ci_metrics tests/e2e/development`、`mypy --follow-imports=skip scripts/ci_metrics`、pre-commit の Ruff・mypy、差分・追跡ファイル確認はすべて通過しました。

PR 作成と、Test ステップの wall time・runner 消費を測る同一条件の Actions 比較（AC-4）は Turn 2 の範囲です。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}