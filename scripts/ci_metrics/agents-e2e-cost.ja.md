# Agent運用e2eの準備コスト

[English](agents-e2e-cost.md) · [简体中文](agents-e2e-cost.zh-CN.md)

[#1524](https://github.com/Motoki0705/tennis-lab/issues/1524) では、メッセージ・親セッション・開発の全ケースを維持して準備を小さくする。
git repository/worktree、fake GitHub、registry、SQLite の書き込み状態は各テストで独立する。
本番コード、CPUの選択条件、shard profile は変更しない。

| 旧ケース・準備 | 維持先・検出する不具合 | 維持する固有の相互作用 |
|---|---|---|
| board の入力・SQL制約・reaction・再試行ごとに確認済みHerdr表示担当を作成 | 同じnodeid/assertを `message_world` で実行。実SQLite transaction、registry/通知lock、fake notifierのdispatchを維持し、表示担当は設定しない | prose/container/fence × post/reply × 外側mentionの全直積、self/retired/unassignedの拒否、atomic性、宛先固定、broadcast意図、配送key、競合、read/ack/reaction状態 |
| broadcast・名簿・代表的なpost/reply/read/ack/rename | 確認済みの稼働Herdr親を持つ既存 `world` | 稼働確認で絞るall、配送と実表示経路、未読数、rename後の同一性。container直積では4番目の親も実際に稼働確認し、誤ったall展開を余分な宛先で検出 |
| boardの下流利用側 | export済み `world` は従来どおり3親の稼働確認を作成 | `test_board_doorbell.py` / `test_board_presence.py` / `test_board_web.py` の表示・socket・HTTPシナリオ |
| 未指示・不正review、割込/同時再送、親停止、準備/起動前失敗回復 | 同じケースを `single_review_env` で実行。実worktree1つ、公開prepare/request、reviewer subprocessを維持 | 不正な権限・入力の全値、起動0回、消費済みrunの耐久性、未消費準備の再試行、競合時も起動1回 |
| 更新/不完全review対象、agent/parse/partial/公開/head競合失敗 | 同じケースを2 PRの `pair_review_env` で実行 | 異なるhead/base、対象の削除/追加、部分coverage、PR側への公開失敗、失敗後の再試行 |
| 指示前head変更・snapshot収集中の変更・peer/base証拠・main追従 | 同じテストを2 PRで準備 | 実git snapshotとhead競合、review済みpeerと残るPR、未review履歴、resolution経路 |
| 大きいdiffをprepare/requestし、拡張後に再びprepare/request | `test_large_diff_is_available_in_full_without_extra_review` で最初の公開要求前に拡張 | 15,000行の全文diff、短いprompt、reviewer1回。古い対象の拒否は上記2つのhead/snapshot競合テストで維持 |
| 取消の6変種: merge済み3子＋取消子13 | 同じ `test_completed_parent_distinguishes_confirmed_cancellation_from_success` の6変種を、merge子10＋取消子11へ縮小 | 実review/gate merge、close/結果確認と取消receiptの併存。missing/body/state-reason/open-PR/workerの各不具合で完了を拒否。3 PRの順序は既存merge-gate/sequential-peerテストで維持 |
| 採用撤回: 出典/結果の設定前後でprepare/requestを反復 | 同じ2ケースで最初のreview要求前に出典/結果を設定 | 親/子の採用撤回、独立3 PRの判断、影響のない兄弟PRのmerge、review1回 |
| fixtureごとに指示済みreviewを再構築 | `ReviewSeed` がworkerごとに1/2/3 PRの各変種を実公開CLIで初回作成し、merge済み `DirectorySeed` で私有ファイルをコピーして実git worktreeをrepair | request・環境もコピーし、git object/ref/index、GitHub/registry/reviewファイルを共有しない。新設の独立性テストは元のrequest・output・GitHub状態を壊し、復元側のreview成功を確認 |

元の679件を全て維持し、seed独立性1件を加える（計680件）。入力分類、post/reply直積、失敗・再試行ケースは削除しない。
取消6ケースのnodeidだけから、冗長なfixture suffix `-independent` がなくなる。`change` の6値は維持する。
共有 `review_env` / `make_review_env` / `review_setup` の既定3 PRは下流向けに維持する。

## 保存済みActionsの監査

run [40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121)、attempt 1 のJUnitは12,782件で、
**12,685 passed / 96 skipped / 1 failed**。失敗は中国語README overviewの数値一致だけだった。
4 shardの `pytest.log` / JUnit / phase eventsに他のfailure/error、重複phase、再実行、worker再起動はない。
中国語の技術用語表記を英語と同じ `2D` / `3D` に揃え、検査自体は変更しない。

| ファイル | 件数 | Setup秒 | Call秒 | Teardown秒 |
|---|---:|---:|---:|---:|
| Board messages | 539 | 43.415 | 545.179 | 0.394 |
| Parent session | 97 | 192.810 | 372.032 | 0.072 |
| Parent development | 43 | 61.148 | 288.234 | 0.029 |

これは過去のrunner観測であり、ローカル時間との前後比較ではない。

## ローカル計測・検証

| 指標 | 前 | 後 |
|---|---:|---:|
| agents e2e件数 | 679 | 680 |
| `test_board_messages.py` setup s | 93.346 | 65.762 |
| `test_board_messages.py` call s | 1136.355 | 181.091 |
| `test_board_messages.py` teardown s | 0.320 | 0.234 |
| `test_parent_session.py` setup s | 298.850 | 26.398 |
| `test_parent_session.py` call s | 603.133 | 431.577 |
| `test_parent_session.py` teardown s | 0.047 | 0.042 |
| `test_parent_development.py` setup s | 22.281 | 16.474 |
| `test_parent_development.py` call s | 526.130 | 295.990 |
| `test_parent_development.py` teardown s | 0.020 | 0.019 |
| 担当3ファイルのphase合計秒 | 2680.482 | 1017.587 |
| 3ファイル＋READMEのコマンド経過秒 | 2695.062 | 1024.909 |
| git commit | 388 | 203 |
| git worktree add / repair | 223 / 9 | 81 / 55 |
| parent CLI | 929 | 549 |
| review / gate CLI | 102 | 103 |
| board CLI | 196 | 196 |
| 実際のfake GitHub起動 | 31986 | 17065 |
| fake Herdr journal行数 | 20025 | 2400 |

担当3ファイルのphase合計の観測上の削減率は **62.04%**。 ローカルphase合計の40%目標を達成。

3種類の欠陥を前後の1 worker版へ一時注入した。コード内mentionの宛先化は10/10件、draft受入れは1/1件、取消検証の省略は2/2件で検出。全runはexit 1・JUnit error 0で、注入関数の実行回数も正である。初期の取消injectorは検証箇所へ到達する前にtimeoutしたため検出証拠から除外し、その記録はJSONに残す。

通常の最新controlは **前723 passed / 後724 passed**。それぞれ **README全44件** を含む。後の追加1件は実review起動でseed独立性を確認する。先行したseed対象の17件、および独立性単独1件も通過。`.venv/bin/python -m spin lint`、変更3ファイルのmypy、Ruff format検査も成功。

## 再現と制限

[計測runner](README.ja.md) に `--workers 1` と担当3ファイル＋README検査を渡す。

```bash
.venv/bin/python -m scripts.ci_metrics run --output /absolute/new/observation --workers 1 -- \
  tests/e2e/agents_ops/test_board_messages.py \
  tests/e2e/agents_ops/test_parent_session.py \
  tests/e2e/agents_ops/test_parent_development.py \
  tests/e2e/development/test_readme_layout.py
```

operation observerはseed外の専用journalで実際のfake GitHub起動を数え、コピーされた準備ログを再計上しない。pytestから直接起動したgit（CLI子孫のgitは含まない）、parent/board/review-gate CLI、
終了時のfake GitHub/Herdr journalを数える。subprocessは変更せず元へ渡す。
[機械可読証拠](agents-e2e-evidence.json) にobserverと一時注入のソース、正確なselector、phase時間、inventory、失敗・中断attemptを残す。

1 worker、native thread 1、CUDA非表示、lock済みCPU環境を揃える。他セッションがWSLホストを共有する。
各比較はn=1の観測でcache条件は不明。Actionsの壁時計短縮や全CPUの中央値/p95は証明しない。
fixture時間はsetupに含まれ、phase合計はCPU時間ではなくテストの経過時間を表す。
profile更新・全CPU inventory/CI性能の確認は統合後に親が行う。本タスクではprofile/workflowを変更しない。

前後とも #1527 を含む更新後mainを使う。seedの初回準備はworker/shardごとに繰り返すため、1 workerのcache hit率から10 shardのCI短縮は推定しない。以前の2 worker観測は別の由来として保持し、この比較に混ぜない。

前後ともREADMEの翻訳修正を含み、「前後」はe2e準備の変更を指す。ローカル実行は担当3ファイルとREADME検査だけに限定する。
