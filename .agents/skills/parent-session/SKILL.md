---
name: parent-session
description: 親Issueの担当・分割・委任・レビュー・引継ぎを進める。説明・起票だけの依頼を除く。
---

# 親Issueを進める

人間の指示と受入条件を起点にする。一つの成果を同じIssueで扱い、前段の結果で次の目標が決まる地点で分ける。同じ成果の調査・実装・検証は並行してよい。直接実装するか、統合担当に成果PRを任せる。

**該当する場面に入ったら、操作前にその行の参照先を読む。** 引数・入力例は `PYTHONPATH=.agents/ops .venv/bin/python -m parent <操作> --help` で確認できる。

| 読む場面 | 参照先で確認すること |
|---|---|
| 手動で引き受ける、再開する、要件・担当が変わる | [担当と合意](references/ownership.md) — 担当の照合、実行根拠、変更の影響 |
| dispatcherから起動された | [自律親](references/dispatch.md) — 予約の受理、共有state、親へ戻す最終結果 |
| 作業を委任する、追加調査や修正を任せる | [ワーカー](references/workers.md) — Issueの粒度、専用worktree、起動・継続・統合 |
| 成果PRを確認する、majorへの対応が終わった | [レビューとマージ](references/review.md) — 正式レビュー1回、修正の受入、マージ条件 |
| 誰かの判断が必要、結果を記録する、親を終了する | [記録と終了](references/finish.md) — 通知先、全体の受入、次回改善 |
| 起動・公開の応答が途切れた、担当やrunが食い違う | [復旧](references/recovery.md) — 最初に照合する記録、同じ操作を再開できる条件 |

担当表示の修復には [Herdr](../herdr-session-label/SKILL.md)、親同士の質問・依頼には [掲示板](../parent-board/SKILL.md) を使う。
