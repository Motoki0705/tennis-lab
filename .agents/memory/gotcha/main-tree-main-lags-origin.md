---
title: "メインtreeのローカル main は自動で pull されず、そこで読む規則・memoryが古いまま残る"
applies_to: all
source: "#1566 の親セッション（2026-10-31）"
created: 2026-10-31
last_verified: 2026-10-31
evidence: "メインtreeで `git fetch origin && git rev-list --count HEAD..origin/main` が1以上なら、ローカル main は遅れている"
---

メインtreeの `main` は、PRのmergeでは更新されない。2026-10-31 には `origin/main` より98コミット遅れていた。セッション開始時に読み込まれる `AGENTS.md`・`CLAUDE.md` と、メインtreeで開く `.agents/memory/SUMMARY.md` も、この古い版になる。古いSUMMARYにあった「子のコミット件名に `[skip ci]`」という判断を読み、#1547 で変わった現在の規則（PRの各pushでCIを実行し、`[skip ci]` を使わない）に反する指示をworkerへ出した。

`origin/main` から作ったworktreeのファイルは新しいので、メインtreeとworktreeで同じパスの内容が食い違う。

**使い方:** 作業前に `git fetch origin` し、SUMMARY・規則・skillは `git show origin/main:<path>` か、`origin/main` から作ったworktreeで読む。メインtreeのファイルは変更しない規則なので、メインtreeで `git pull` するかは人間に任せる。
