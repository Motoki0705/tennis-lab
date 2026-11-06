---
title: "Codex workerのpromptで途中終了を数値なしで許すと、時間が残っていても早々に途中終了する"
applies_to: all
source: "#1234（2026-10-16 ball cycleの5 worker）、2026-11-05 PRでballのpromptを修正"
created: 2026-11-05
last_verified: 2026-11-05
evidence: "#1234 本文と、src/tasks/ball_detection/annotation/review/worker-instructions.md の途中終了の条件。playerの worker-instructions.md には同じ形の許可（Defer uncertainty or unmet policy）が残っている"
---

注釈レビューのpromptが「不確かなときや予算切れなら途中で止めてよい」とだけ書き、時間上限を数値で示さず、handoffも先頭64 frameだけだった。5件のworkerすべてが最初の応答で「終わらなければdeferredにする」と宣言し、4時間のtimeoutに対して10〜21分・64〜192/780 frameで止まった。文脈あふれ・timeout・service障害は原因ではなかった。

**使い方:** Codex workerのpromptでは、作業単位（全体）と時間上限を数値で示し、途中終了してよい条件を具体的に列挙する。全体の進捗と次に処理する範囲を毎回見せる。判断できない個別の項目は、その状態を記録して先へ進ませる。
