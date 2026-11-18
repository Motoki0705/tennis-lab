# #2253 の作業指示（親 #1734）

issue #2253 の本文が仕様です。まず読んでください。補足は次のとおりです。

- このworktreeは、#1752のbranch `codex/video-batch-cli`（HEAD d392dd375）から作った。一括CLI、共有court_sideのload設定、3clipの保存済みstoreの場所は、`knowledge/runs/run-i1752-shared-fp16-20261118/`（reproduce.md・full69-command.txt・run.json）を参照する。
- 出力はOUTPUT配下に置く。最終報告に、その絶対pathを書く。親は、その出力からartifactを作る。npzの形式は、issueの「出力」の節のとおりにする。
- 速さを優先する。人間はこの比較を早く見たい。モデルの数は4〜6で、checkpointの取得に時間がかかるものは避けてよい（避けた理由は記録する）。3clipとも短い（235〜356frame）。
- CUDAのハングかエラーが起きたら、再試行しない。全threadのstackとGPUの状態を保存して停止し、報告する。
- 完了したら、knowledgeを記録し、PRを作ってCIを確認し、Ready for reviewにして終了する。merge・closeは親が行う。

### 基準とするbranchについて（親の判断）
このbranchは、#1752の `d392dd375` に固定する。#1751（PR #2182）の新しいレビュー修正は取り込まない。取り込むと各段のcache identityが変わり、3clipの保存済みstoreを読み込めなくなるため。依存関係は委任の記録から外したが、PRのbaseは `codex/video-batch-cli` にする。この比較は、評価の結果（knowledgeとスクリプト）を残すことが目的で、コードをmainへmergeすることは前提にしない。
