---
title: "spin setup-nht は、実行したtreeの third_party/nht に全体共通の nht-render を結び付ける"
applies_to: all
source: "nht-render の ModuleNotFoundError 調査（2026-10-16）"
created: 2026-10-16
last_verified: 2026-10-16
evidence: "`uv tool dir` 配下の nht 環境にある `__editable___nht_*_finder.py` の MAPPING が、どのtreeの third_party/nht を指しているかを見る"
---

`spin setup-nht` は、`uv tool install --editable third_party/nht` で、ユーザー全体に1つの `nht-render` / `nht-reconstruct` を入れる。editable なので、実行したtreeの `third_party/nht` と `third_party/nht/gsplat` をそのまま参照する。

worktreeで実行すると、全体の `nht-render` がそのworktreeに結び付く。worktreeが消えると、どこから実行しても `ModuleNotFoundError: No module named 'nht_pipeline'` で落ちる。2026-10-16 には、削除済みのworktreeを指していて壊れていた。

**使い方:** `spin setup-nht` はメインtreeでだけ実行する。worktreeでNHTを変更して試すときは、全体の tool を入れ替えない。tool の Python に `PYTHONPATH=<worktree>/third_party/nht` を付けて `-m nht_pipeline.render` を実行する、`nht-render` という名前のラッパーを作業用ディレクトリに置く。その絶対パスを `NHTRenderCommandRequest.executable`（reveal なら `reveal.nht.executable`）に渡す。gsplat を変えていなければ、CUDA拡張はメインtreeのものがそのまま使える。
