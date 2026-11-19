# #2269: court_sideをプレーヤーの観測だけから判定する経路（親 #1734）

Issue #2269の本文が要件の正本です。必ず読んでから始めてください。

- branch `codex/court-side-player`、worktree `.claude/worktrees/court-side-player`。基準はorigin/mainの最新。
- 期限: 人間は、2026-11-19 15:00（JST）までに#1757のbenchmarkを動かせる状態を求めています。#1757の担当は、この経路の設定（`court_side.evidence: player`）を使います。
- 作業の順序:
  1. 設定キーと依存グラフの切り替え、判定の中心、unitテストを作る。14:10までにpushする。
  2. pushの時点で、PR（Draftでよい）を開く。PR本文に、設定の名前と使い方を書く。#1757の担当がmergeして使えるようにする。
  3. そのあとで、meiji_4camの3clipの実データでの検証（clipごと・動画共有）を行い、knowledgeに記録する。人物系は、#1752の3clip storeの保存済み成果物を使ってよい（`[local path]`）。
- 設計の方向はIssueに書いたとおりです。人物の同一性を使わず、足元の地面座標の点集合を、仮説ごとに対応付けなしで比べます。実データで判定がうまくいかない場合は、黙って別の方法へ切り替えず、何が起きたかを報告してください。
- GPUを使う場合は、共有training queue経由にしてください。CUDAのエラーやハングが起きたら、再試行せずに止めて報告してください。
- 最後に、PRのURL・HEAD・3clipでの判定の結果（各仮説のscoreとmargin）・CIの結果を報告してください。PRはReadyにしてください。
