# PR #2240 第3回レビューのmajor指摘への対応（親 #1734 から #2238 へ）

前回の修正（公開の再開時に精度を照合する）は、受け入れられた。新たに、knowledgeの再現性について、major指摘が1件出た。

## 固定入力での再計測が、別worktreeのDINOバイナリに依存する
smokeと全尺計測の `repro.sh` は、別worktree `video-batch-cli` の `outputs/` にある `MultiScaleDeformableAttention.so` を、絶対パスで指定している。取得とビルドの手順がない。knowledge/README.mdの再現要件に沿って、固定した版とビルド条件から再生成し、SHA-256を照合できる手順を追加すること。

親の方針: **直す（ドキュメントと再現スクリプトだけ）。** 別worktreeを参照するのをやめる。ビルドの手順（#1752のknowledgeにある `rebuild-native.sh` と同じ方法でよい）と、ビルド条件（CUDA、torch、コンパイラの版）を記録する。再生成したバイナリのSHA-256を照合する手順にする。GPUの再計測はしない。CI通過後、PR本文に追記する。merge・closeは親が行う。
