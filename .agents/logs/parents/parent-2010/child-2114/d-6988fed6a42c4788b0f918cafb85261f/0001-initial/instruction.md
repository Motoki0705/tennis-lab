# #2114 の実装（PR作成まで）

担当Issue #2114 の本文が要件と受入条件の正本です。親の実測は https://github.com/Motoki0705/tennis-lab/issues/2013#issuecomment-6890260495 にあります。

## 背景（実測）

2026-11-15、上限が戻った直後に4つの親の merge gate が同時に動き、REST の共有上限（5000回/時）を毎分150〜250回のペースで使いました。1回の merge gate は13分たっても終わらず、#1968・#1893・#1900 はそれぞれ約750〜800回を呼びました。#2010 の実行（約5分）の内訳は次のとおりです。

| endpoint | 回数 |
|---|---:|
| `issues/N/comments?per_page=100&page=N` | 226 |
| `issues/N` | 202 |
| `issues/N/parent` | 149 |
| `issues/N/sub_issues?per_page=100&page=N` | 149 |
| `pulls/N` | 70 |
| `issues/N/timeline?per_page=100&page=N` | 68 |
| `issues?state=all&labels=parent-evidence&per_page=100&page=N` | 30 |

読んだ Issue は約60件で、自分の家族以外（#1893・#1900・#1990 など他の親とその子・証拠用Issue）を含みます。#2010 自身は38回読まれていました。`review/merge_gate.py` の `collect()` は、収集の全体を `external.gh.fresh_reads()` で包んでいるので、同じ実行の中でも GET cache が効きません。

## 進め方

1. まず fake GitHub（`tests/benchmarks/agents_ops/parent_http.py` や `tests/e2e/agents_ops` の既存の fake gh）と `PARENT_METRICS_FILE` で、merge gate 1回の endpoint 別の呼び出し回数を、親・子・PR・証拠用Issue・他の親が複数ある現実に近い状態で計測する。どの関数が何回読んでいるかを特定し、PR本文に表で載せる。
2. 削減する。
   - 1回の観測の中では同じ endpoint を1回だけ読む。`collect()` 全体の `fresh_reads()` をやめ、観測の始まりで1回 fresh にしたうえで、その観測の中では cache を使う形を検討する（#2029 の契約: cache は transaction epoch の中だけ、`detached()` は cache なし）。
   - merge 直前の確認（head・base・レビュー・CI・担当・採用・依存）は、明示的に fresh に読み直す。ここは削らない。
   - 他の親の Issue・履歴を読む経路（証拠用Issueの親照合、レビュー登録の重なり確認、`check_task_adoptions`、依存の確認など）が本当に必要かを確かめ、必要な情報だけを必要な範囲で読む。必要な理由が判定の安全性にある経路は残し、理由をコメントかテストで明示する。
   - 判定条件を弱めない。読み取りを減らした結果、変化を見落とす経路を作らない。静かなフォールバックを作らない。
3. 必要なら同じ考え方を `review/parent_review.py` の資料準備と事後照合にも適用する（同じ実測で `parent_review` も呼び出しが多い）。範囲が大きくなる場合は merge gate を優先し、残りを報告する。
4. 回帰テスト: 呼び出し回数の上限を固定するテスト、他の親の Issue を読まないテスト、収集の後・merge の前に head・レビュー・CI・担当が変わったら merge しないテスト。テストは設計してから書く。
5. 関連テストと pre-commit を通し、`Closes #2114` 付きでPRを作り、対象HEADの CI（Python）の成功を確認する。

## 注意

- 共有の GitHub API 上限は今も逼迫しています。計測と検証は fake GitHub で行い、実 GitHub は push・PR作成・CI の確認だけに使ってください。`gh pr checks --watch` などの短い間隔の polling はしないでください。CI の待機は数分おきの確認にしてください。
- 並行して #2040 の PR #2054（`parent/github.py`・`issues/cli.py` の通信層を HTTP に置き換え）と #2031 の PR #2056（dispatcher・triage）が開いています。これらのファイルの変更は必要最小限にし、衝突したら後から merge する側が rebase します。
- 時間の上限は6時間。止めてよいのは、(1) 判定の安全性を保ったまま読み取りを減らせない箇所があり、親の判断が要る、(2) 6時間を超えた、のどちらかだけ。止める場合は、済んだ範囲・未了の受入条件・次の一手を報告する。
