---
title: "メインcheckoutの定期同期はlockや作業中の変更で延期される"
summary: "メインcheckoutは30分ごとに更新されるが、運用jobのlockや作業中の変更などで遅れ得る。参照する版をorigin/mainと照合する。"
applies_to: all
source: "#1566 の親セッション（2026-10-31）"
created: 2026-10-31
last_verified: 2026-11-20
evidence: "timerの稼働状態と git rev-list --count HEAD..origin/main を比較する。延期条件と状態の見方は git_sync README にある。"
citations:
  - path: ".agents/ops/git_sync/systemd/tennis-lab-git-sync.timer"
    symbol: "OnCalendar"
  - path: ".agents/ops/git_sync/README.md"
    symbol: "#更新の契約"
  - path: ".agents/ops/git_sync/README.md"
    symbol: "#運用jobとの排他"
---

メインtreeの `main` は `tennis-lab-git-sync.timer` が30分ごとにfast-forwardする。実行間隔のほか、共有checkout lockを運用jobが保持していたり、追跡ファイルの変更や進行中のGit操作があったりすると更新が遅れる。2026-11-20の確認ではtimerが稼働中でもメインtreeは22コミット遅れていた。

**使い方:** 別の版を疑ったら `git fetch origin` と `git rev-list --count HEAD..origin/main` で確認する。運用の正本を参照するときは `git show origin/main:<path>` か、そのrefから作ったworktreeで読む。現在の延期理由は `spin ops status git-sync` で確認する。
