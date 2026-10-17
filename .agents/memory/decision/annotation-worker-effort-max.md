---
title: "注釈workerのeffortはmaxに固定する"
applies_to: all
source: "2026-10-16 ユーザー決定、#1233（2026-10-17 再確認）"
created: 2026-10-17
last_verified: 2026-10-17
evidence: "#1233 と後続のユーザー指示を確認し、注釈workerのmodel・effortを変更する決定がないか照合する"
---

ユーザーはローカル注釈worker（gpt-6.1-sol）をmaxで動かすと決定した。2026-10-15以降のplanにあるhighはユーザーの判断ではなく、実装時のテスト値が混入したもの。

**使い方:** 新規planには決定済みのmodel・effortを使う。過去planの値をユーザー承認の根拠にしない。既存campaignの不変planは書き換えない。
