---
title: "merge前にCIを起動せず、CIをmergeの条件にしない"
applies_to: all
source: "#1388（2026-10-23のユーザー指示「マージについてCI gateは不要です。そもそもCIをマージ前に起動しないようにしてください。」と、今後の親作業でも既定にするという回答）"
created: 2026-10-23
last_verified: 2026-10-23
evidence: "ユーザー指示に基づく判断。#1428・#1429・#1430 はコミット件名の `[skip ci]` でPR時のCIを起動せずにmergeした。.agents/ops/review/required_checks.txt が残る間は merge gate がCIを必須にする"
---

ユーザーは、PR作成・pushのたびにCIを走らせることも、CIの結果をmergeの条件にすることも求めていない。CIはmerge後のmainでだけ走ればよい。

`.github/workflows/ci.yml` は `pull_request` と main への `push` で起動するため、子のコミット件名の末尾に `[skip ci]` を付けると、PRとbranchのpushではCIが起動しない（`pull_request_target` のlabelerは動くが、CIではない）。merge commitの件名には付かないので、merge後のmainではCIが走る。

**使い方:** 親が子へ実装を委任するときは、全コミットの件名末尾に `[skip ci]` を付けるよう指示する。検証はローカルのpytest（共有lock経由）・ruff・mypyで行い、PR本文に記録する。merge gateは `required_checks.txt` のCIを必須にしているので、その仕組みが変わるまでは、親単位レビューと指摘の解消を確認した後、レビュー済みheadを `gh pr merge --merge --match-head-commit <head>` に渡して依存順にmergeする。merge後のmainのCIは確認し、失敗が自分の変更によるものか、main側の既存の失敗かを切り分けて報告する。
