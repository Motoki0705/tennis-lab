---
title: "cleanup集計と完全明細の可用性が一致しなかった"
applies_to: "all"
source: "https://github.com/Motoki0705/tennis-lab/issues/1280"
created: 2026-10-19
last_verified: 2026-10-19
evidence: "当該レポートの収集データcleanup.report_json、.agents/ops/ops-report/opsreport/collect.py:119-122,477-489、.agents/ops/ops-report/opsreport/report.py:65-68,85-86。固定版beba5f9a94d598efe52cf7606a9cbdebae6e2f5d。"
---

**注目領域:** 運用レポートの入力完全性

2026-10-19の収集結果はcleanup.status=okだが、report_jsonに5162348文字の省略表示があった。集計は確認できたものの、その文字列だけでは全候補の明細を復元できなかった。約102.6 GBは要承認区分の集計であり、削除可能容量を実証した値ではない。

**未解決仮説:** 同じ切り詰め方式を続けた場合、次回も候補数に応じて明細が欠落する可能性がある。次回の実際の入力サイズは未測定。

**見送り理由:** なし

**次回の再検証条件:** 次回レポートでcleanupの完全性を確認する。collector・入力schema・artifact受渡しを変更した場合は、大きな結果でも完全明細を参照できるか再検証する。

**参照:** 運用レポート 2026-10-19
