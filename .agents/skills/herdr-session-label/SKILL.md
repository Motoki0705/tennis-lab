---
name: herdr-session-label
description: HerdrのAgent一覧へ、自分の名前・担当する親issueの番号・役割・短い題名を同期する。親担当の開始・再開・引継ぎ・終了、表示の修復に使う。担当の割当や他セッションの一括変更は行わない。
---

# Herdrに担当を表示する

担当の正本はparent registry。仕様・ホスト設定・失敗時の扱いは
[表示CLI README](../../ops/herdr_labels/README.md)を参照する。
`parent-session`による担当登録・引継ぎを済ませてから表示する。
会話に現れたissue番号や古いターミナル題名から担当を推測しない。

parent CLIの`register` / `reconcile` / `sync` / `handoff` / `release`が同期する。
結果の`display_sync`を確認し、`error`を放置しない。
`parent_action_applied: true`なら担当操作は適用済みなので、表示だけを再試行する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m herdr_labels sync
# 親タイトルの識別名では長い・分かりづらいときだけ、短い題名を指定する
PYTHONPATH=.agents/ops .venv/bin/python -m herdr_labels sync --parent 1353 --topic セッション表示
```

自分の実行環境のセッション識別子とHerdr paneを使う。他セッションのIDを設定して実行しない。
通常はHerdr環境変数が必要。欠落する実行基盤では、READMEの明示bind手順で
自分の会話であることを照合する。フォーカスや一覧順から対象を選ばない。
`not_configured`は表示未設定であり、成功ではない。

Claudeの起動・再開・終了hookは表示の復元・解除だけを行う。担当登録・解除は親の明示操作を使う。
終了後に表示だけ残った場合は、自分のセッションから`python -m herdr_labels clear`を同じ環境で実行する。
色・稼働状態の更新、プロセスの起動・停止、他のpaneの操作はこのskillの範囲外。

他セッションの初回移行は、人間が一括変更を明示依頼したときだけ親が実施する。
実担当をローカル記録と会話で照合し、親・関連作業・親未登録を区別する。
ローカル識別子やセッション対応表を公開issue/PRへ貼らない。
