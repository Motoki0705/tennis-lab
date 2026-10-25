# merge fixture準備の削減

[English](fixture-cost.md) · [简体中文](fixture-cost.zh-CN.md)

[#1402](https://github.com/Motoki0705/tennis-lab/issues/1402) では、1 PRの判定に必要な
fixtureを1 PRに縮めます。独立/依存順、部分判定、base前進、失敗・不確実merge、修正公開、
head競合、新規PRとの相互作用は3 PRと実git worktreeを維持します。変更可能なgit・GitHub
状態・review記録はcaseごとに分離したままです。

| 旧準備 | 維持先・契約 |
|---|---|
| 拒否・dry-run・単一対象でも毎回3 PR | `review_setup(pr_count=...)` と `make_gate_env` で必要数を選ぶ。人間保留、最新review/head/diff、fork/draft/conflict、agent未起動を維持 |
| review準備後に依存・branch・変更pathを変更し再準備 | `before_review` で最初の公開要求前に確定。実CLIとgit snapshotは残す |
| auto-merge異常4値ごとにgit/合意/3 PR/review | 4値のpolicy検査と、I/O・state作成より前に拒否する実CLI 1件へ分離 |
| 親の入力異常・回復でも常に3 PR | 対象欠落の相互作用は3、独立した異常・回復は1。公開・identity・回復契約を維持 |

対象は `test_merge_gate.py` と `test_parent_session.py`（`tests/e2e/agents_ops/`）。
mainで廃止された変更pathによる追加承認は復活させず、通常レビュー・人間保留・実mergeのopt-inを維持します。

## 検証・実測

変更2ファイルは259件すべて成功しました。異常値4件をpolicy 4件＋CLI 1件へ分けたため
全体では1件増えています。テスト数ではなく準備コストの削減です。
auto-merge不正値を許す変異は旧4/新5件、人間保留を無視する変異は旧3/新3件で検出しました。
4実行とも終了値1の通常assert失敗で、setup/収集エラーはありません。

| 代表サンプル | 旧 | 新 |
|---|---:|---:|
| 成功件数 | 8 | 9 |
| setup合計秒 | 78.155 | 26.636 |
| call合計秒 | 16.729 | 9.350 |
| コマンド単調時計秒 | 58.828 | 23.558 |
| gate fixture数 | 8 | 4 |
| git commit / worktree作成 | 40 / 24 | 14 / 6 |
| 親CLI / fake GitHub呼び出し | 82 / 3391 | 34 / 1255 |
| review prepare / request | 9 / 9 | 4 / 4 |

2 worker、native thread各1、CUDA不可視、同じ依存での各1回の部分観測です。
他のローカル検証も動いており、全CIやGitHub runnerの改善率とは扱いません。複数PR相互作用は
3 PRのままです。[機械証拠](fixture-evidence.json)に版・結果・phase・回数・変異を保存します。
代表ケースと再現selector、回数observerの手順は[英語版](fixture-cost.md#reproduction)を参照してください。
