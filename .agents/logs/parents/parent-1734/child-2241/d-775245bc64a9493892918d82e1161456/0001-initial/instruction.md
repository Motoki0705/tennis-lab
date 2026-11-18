# #2241 の作業指示（親 #1734）

issue #2241 の本文が仕様です。まず読んでください。補足は次のとおりです。

- 手本は#2191（PR #2222、merge済み）の `src/tennis_scene/pipeline/video_court/` です。build / load の明示的な選択、content-addressedな保存、drift診断の分離、`unobservable_policy` の考え方にそろえてください。
- 実データの確認には、meiji_4camの保存済みのball・校正の成果物を使います。全clip分がそろっていない場合は、必要な段（Court→校正→Ball）だけを、共有training queue経由で再計算します。共有コートのstoreとdigestは、branch `codex/video-batch-cli` の `knowledge/runs/run-i1752-doubles-20261118/full69-command.txt` にあるものを使います。
- 参考: clip_037とclip_027は、単独の判定で最良の仮説が [F,F,T,T] で、marginは0.130と0.116（閾値0.15）。clip_000は単独で決まった。
- CUDAのハングかエラーが起きたら、再試行しない。全threadのstackとGPUの状態を保存して停止し、報告する。
- 完了したら、PRを作ってCIを確認し、Ready for reviewにして終了する。merge・closeは親が行う。
