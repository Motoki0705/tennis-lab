---
name: research-loop
description: Run an issue-driven research loop - a theme issue with human-agreed metric and stop conditions, then autonomous survey/select/experiment/discuss cycles as child issues.
---

# Research Loop

研究テーマ＝親issue、1サイクル（調査→選定→実験→考察）＝子issue。
人間が介入するのはテーマ設定だけで、以降はAIが停止条件まで自律で回す。
途中経過は週次運用レポートで観察される。この手順自体も仮説であり、各サイクルの
「skill改善メモ」をもとに改訂する。

## 1. テーマを固定する（人間と合意）

[theme.md](references/theme.md) に従い、[テーマtemplate](../../../.github/ISSUE_TEMPLATE/04_research_theme.md)で
問題提起・目標指標・評価条件（split・seed数）・停止条件・予算の目安を合意して
issue本文に固定する。共通の収束ルールは無い。指標の性質から停止条件をAIが提案する。
合意前に実験を始めない。1サイクルで止まるテーマ（調査issueを1回検証して報告する
従来の運用）もこの最小形として同じ手順で扱う。

## 2. サイクルを回す（AI自律）

[cycle.md](references/cycle.md) の順に、[サイクルtemplate](../../../.github/ISSUE_TEMPLATE/05_research_cycle.md)の子issueを
1つ起こして進める。

- 調査: [literature.md](references/literature.md)（`scripts/lit_search.py` と新規性の観点）と
  [knowledge summary](../../../knowledge/summary.md)。
- 実験: GPUを使うものはすべて [training-queue](../training-queue/SKILL.md) 経由。
- 記録: 実測は [knowledge-control](../knowledge-control/SKILL.md) でノード化し、
  主張は [evidence-checklist.md](references/evidence-checklist.md) で確かめる。
- 結論: 考察・停止条件の判定・次サイクル提案・skill改善メモを子issueに書く。

固定した評価条件はサイクル内で変えない。変える必要が出たら、テーマissueで人間と
再合意してから次サイクルに反映する。

## 3. 止める

停止条件を満たしたら、テーマissueに最終報告をコメントして止まる。
テーマissueのcloseは人間が行う。

ARIS（PR #1033）由来の部品の出典とライセンスは [LICENSE.ARIS.txt](LICENSE.ARIS.txt)。
