---
title: "coverageの異なる方式比較は共通・追加・喪失の誤差を分ける"
applies_to: all
source: "2026-11-04: #1554の明示レビューcross-1（PR #1672 / #1674 / #1676）、人間と親が妥当と判断"
created: 2026-11-03  # UTC: 作成時刻はJSTの11月4日
last_verified: 2026-11-03
evidence: "knowledge/runs/run-ball3d-dense-likelihood-20261103/cohorts.pyとcohort-receipt.jsonで集合の件数再構成・元の採点値一致を確認する。将来READMEへ昇格したら重複を削除する。"
---

人間は、coverageが変わる方式比較で全採点点と共通点だけを示しても、増分の品質を判断できないとした。
採点数の純増は追加点と喪失点の差なので、同じ純増でも対象点と誤差の分布が異なる。

**使い方:** 同じ採点器の標本ID（ball3dなら除外cameraとframe）を突き合わせ、
共通点は両方式、追加点は新方式、喪失点はbaselineの件数と誤差分布を区間別に併記する。
共通＋追加＝方式の全採点数、共通＋喪失＝baselineの全採点数を検算し、空集合は未測定とする。
