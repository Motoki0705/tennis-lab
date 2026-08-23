---
id: group-i786-normalization-v1-v2
type: group
title: court座標正規化 v1/v2 baseline比較 (#786)
issue: 786
members:
- run-i786-plcs-norm-v1
- run-i786-plcs-v2-resume-b24-r2
- run-i786-blcs-norm-v1-b64-w16
- run-i786-blcs-norm-v2-b64-w16
parents: []
tags: [normalization, blcs, plcs, issue-786]
---

## まとめ

Issue #786のcourt座標正規化v1/v2 baseline群。BLCSは同一model・batch 64・worker 16・100 epochで制御比較し、v2のposition errorが2.8%低下した。PLCSはv2継続時にbatchを4から24へ変更したため完全な一変数比較ではないが、position errorは低下し、Z誤差とrotation誤差は増加した。いずれも単一seedであり、採否の因果証拠ではなく実装後baselineとして扱う。
