---
name: parent-session
description: 親Issueの担当・分割・委任・レビュー・引継ぎを進める。説明・起票だけの依頼を除く。
---

# 親Issueを進める

親が引き受けるのは、合意した成果を受入条件に照らして届けるところまで。まず依頼者の指示、親Issueの本文と最新コメント、担当、既存のPR・ワーカーを確認する。人間がPRのレビューやマージを引き受けた場合は、その地点で成果と残件を渡す。

手動親はrepoのcheckoutで次のように現在地を調べる。以下の `parent` はこのCLIを指す。共通引数は操作名の前に置く。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 status
```

**必要な場面で対応する参照先を読む。** 初手は手動親なら「担当と合意」、dispatcherから起動された親なら「自律親」。以降は現在の操作に進む。引数・入力ファイルの形は `parent <操作> --help` で確認する。

| 場面 | 参照先で判断できること |
|---|---|
| 手動の担当開始、再開、担当・要件の変更 | [担当と合意](references/ownership.md) — 誰が何を実行できるか、既存の合意・証拠をどこまで使えるか |
| dispatcherから予約を受けた | [自律親](references/dispatch.md) — どのstateで受理するか、いつ待機・完了を報告するか |
| 作業を分ける、調査・実装・追加検証を任せる | [ワーカー](references/workers.md) — Issueと担当の選び方、専用worktree、待機、成果の統合 |
| 成果PRの準備ができた、レビュー結果が届いた | [レビューとマージ](references/review.md) — 1回の正式レビュー、majorの受入、CI待ちと親判断 |
| 人間への質問、判断・成果の記録、担当終了 | [記録と終了](references/finish.md) — 通知先、完了と取消の区別、全条件の受入、次回改善 |
| CLIが非ゼロ終了した、応答や元のrunが見つからない | [復旧](references/recovery.md) — 保存済みの何を照合し、どの操作から再開するか |

一つの成果なら同じIssueで直接実装するか、統合担当を決めて並行作業を委任する。前段の結果で次の目標が決まる地点が子Issueの境界。成果PRの正式レビューは1回。親がmajorの対処を確認したら、マージ条件の照合へ進む。

担当表示の修復は [Herdr](../herdr-session-label/SKILL.md)、親同士の連絡は [掲示板](../parent-board/SKILL.md) を使う。
