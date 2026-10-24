---
title: "merge前にCIを起動せず、CIをmergeの条件にしない"
applies_to: all
source: "#1388（2026-10-23のユーザー指示「マージについてCI gateは不要です。そもそもCIをマージ前に起動しないようにしてください。」と、今後の親作業でも既定にするという回答）、#1349 方針7・#1449"
created: 2026-10-23
last_verified: 2026-10-24
evidence: "#1349 方針7のユーザー合意と #1449。.agents/ops/review/review_policy.py の evaluate_gate がCIを判定入力に含めず、merge_gate_policy.py がCIを観測値として記録することを確認する"
---

ユーザーは、PR作成・pushのたびにCIを走らせることも、CIの結果をmergeの条件にすることも求めていない。CIはmerge後のmainでだけ走ればよい。

`.github/workflows/ci.yml` は `pull_request` と main への `push` で起動するため、子のコミット件名の末尾に `[skip ci]` を付けると、PRとbranchのpushではCIが起動しない（`pull_request_target` のlabelerは動くが、CIではない）。merge commitの件名には付かないので、merge後のmainではCIが走る。

**使い方:** 親が子へ実装を委任するときは、全コミットの件名末尾に `[skip ci]` を付けるよう指示する。検証はローカルのpytest（共有lock経由）・ruff・mypyで行い、PR本文に記録する。親単位レビューと指摘の解消を確認した後、[正規のmerge gate](../../ops/review/README.md#merge-gate) で依存順にmergeする。CIの有無・未完了・失敗では待たない。merge後のmainのCIは確認し、失敗が自分の変更によるものか、main側の既存の失敗かを切り分けて報告する。専用memory PRのCI要件は [限定経路](../../ops/ops-report/README.md#専用memoryの保存) の合意を維持する。
