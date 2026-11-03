---
title: "WSL2 では親レビューの隔離CPUテストのprobeが通らず、CIログも読めないので、指示に書かないと統合レビューが `incomplete` で終わる"
applies_to: all
source: "#1581 の PR #1636 の統合レビュー（2026-11-02）"
created: 2026-11-02
last_verified: 2026-11-03
evidence: "`PYTHONPATH=.agents/ops/ops-report .venv/bin/python -B -m opsreport.reproduction_server probe --directory <新しい空dir>` の `probe.json` で全challengeが `scope cleanup could not be verified` になる。CIログは #1609 が閉じるまで `ci_log` が失敗する"
---

このWSL2環境では、親レビューの隔離CPUテスト用のprobe（`opsreport.reproduction_server probe`）が、5つのchallengeすべてで `scope cleanup could not be verified: tennis-audit-cpu-<id>.scope` になって通らない。有効なproofが作れないので、reviewerに `cpu_tests` を提供できない。さらに、#1609 が閉じるまでは、`ci_log` がgh 2.103のエスケープシーケンス拒否でCIログの本文を返さない。

reviewerはこの2つを「取得失敗・未確認」と扱い、静的な確認でblocker/majorが無くても、PRを `not_reviewed`（verdict `incomplete`）にする。merge gateは `not_reviewed` のPRを止めるので、通常どおり指示しただけでは何度レビューしてもmergeできない。2026-11-02には、2回続けて `incomplete` になった。

**使い方:** この環境で `review-request` を出すときは、指示文に、CIログの本文と隔離CPUテストが使えないこと、対象HEADのcheck runの結論はAPIで確認できることを書く。そのうえで、この2点だけを理由に `not_reviewed` にせず、固定commitのコード・差分全文・issueの証跡・CIの結論で判定するよう求める。ログ本文やテストの再実行がなければ判断できない点が具体的にある場合だけ、その点を明記して `not_reviewed` にさせる。2026-11-02（PR #1636）と2026-11-03（PR #1668）には、指示なしで `not_reviewed` になり、この指示で `reviewed` / `approve` になった。#1609 が直り、probeが通る環境になったら、このエントリを削除する。
