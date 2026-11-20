---
title: "Codex workerのpromptで途中終了を数値なしで許すと、時間が残っていても早々に途中終了する"
summary: "Codex注釈workerに曖昧な途中終了条件を渡すと短時間で止まった例がある。作業範囲・時間上限・終了条件を具体化する。"
applies_to: all
source: "#1234（2026-10-16 ball cycleの5 worker）、2026-11-05 PRでballのpromptを修正"
created: 2026-11-05
last_verified: 2026-11-20
evidence: "#1234 本文と、src/tasks/ball_detection/annotation/review/worker-instructions.md の途中終了の条件。playerの worker-instructions.md には同じ形の許可（Defer uncertainty or unmet policy）が残っている"
citations:
  - path: "src/tasks/ball_detection/annotation/review/worker-instructions.md"
    symbol: "against the time limit"
  - path: "src/tasks/player_detection/annotation/review/worker-instructions.md"
    symbol: "Defer uncertainty or unmet policy"
---

注釈レビューのpromptが「不確かなときや予算切れなら途中で止めてよい」とだけ書き、時間上限を数値で示さず、handoffも先頭64 frameだけだった。5件のworkerすべてが最初の応答で「終わらなければdeferredにする」と宣言し、4時間のtimeoutに対して10〜21分・64〜192/780 frameで止まった。文脈あふれ・timeout・service障害は原因ではなかった。

**使い方:** 全件の確認を任せるときは、作業単位と決まっている時間上限、途中終了の条件を明示する。全体の進捗と次に処理する範囲を示す。個別の判断不能は記録して作業を進める。
