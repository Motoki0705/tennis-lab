# #1755: clip・cameraをまたいだbatch推論とメモリ上限（親 #1734）

Issue #1755の本文が要件の正本です。必ず読んでから始めてください。

- branch `codex/video-batch-batching`、worktree `.claude/worktrees/video-batch-batching`。#1752の一括CLI（`codex/video-batch-cli`、HEAD 8ab460050）の上に積んでいる。
- まず `origin/main` をmergeする（#2225と#2226がmainに入っている。#2226がまだmerge中なら、`origin/codex/person-feature-artifacts` をmergeする）。親が試したところ、#2226のbranchとは衝突があった。両方の意味を保って解くこと。
- 2026-11-19の範囲の変更: video全体の本番処理ではball検出を外し、3D再構成はプレーヤーだけにする。そのため、batch adapterは人物系（DINO・ViTPose・CLIP・HMR2）とCourtを優先する。Ballのadapterは、#1757のbenchmarkがball推論を含むので必要だが、後回しにしてよい。後回しにした場合は、その旨をPRに明記する。
- #1757（3clipのbenchmark）が、この実装の「モデル再利用＋batch化」の条件を計測する。早い段階で動く版をpushし、PR（Draft）に使い方（設定のキー）を書くこと。
- OOMのときにbatchを下げる、CPUへ移すといった暗黙の代替処理は作らない。
- GPUを使う場合は共有training queue経由にする。CUDAのエラーやハングが起きたら、再試行せずに止めて報告すること。
- 最後に、PRのURL・HEAD・実装したadapter・テスト・CIの結果を報告し、PRをReadyにすること。
