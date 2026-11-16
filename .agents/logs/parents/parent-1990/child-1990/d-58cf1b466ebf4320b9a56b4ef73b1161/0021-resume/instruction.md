# #1990 追加修正: 使用中判定の一時的な [local path] 読取拒否で cycle が失敗する

PR #2148 は merge 済み、実機の配置切替・受入・sudo保守も完了した（親が実施）。user timer `tennis-lab-storage-tier.timer` も main のコードで導入・有効化した。残る問題を、新しいPRで直してほしい。

worktree `.claude/worktrees/storage-tiering` は、親が `codex/storage-tiering` を最新の origin/main（10d379f23）へ付け替えた。このbranchから新しいPRを作ること（PR本文は日本語、閉じ指示は `Refs #1990`。親がレビュー前に `Closes` へ変える）。旧PR #2023・#2148 は merge 済みで、触らない。

## 現象（実機）
- 11/16 の移行中に hydrate 2回・verify 1回、11/17 に user service の単発 cycle 4回中1回が、`usage cannot be verified: pid=…: [Errno 13] Permission denied: '[local path]'; maps is readable; nondumpability is unproven; …` で失敗した。失敗は安全側（hydrate の stage は破棄、cycle はその単位だけ failed で exit 1）で、再実行すると成功する。
- 原因の候補（親の観察）: 同じuidで短命なプロセスが頻繁に exec する（例: ユーザー端末の `watch -t nvidia-smi` が2秒ごとに `sh -c nvidia-smi` を起動）。exec の遷移中や終了直前に `[local path]` だけが EACCES になり、`maps` は読めるため nondumpable の証明も立たず、`.agents/ops/storage_tier/usage.py` の `inspect` が unknown にする。親の30秒のサンプリングでは、fd を読めなかった同uidプロセスは nvidia-smi（exeも直後に消失）、sshd、gpg-agent、(sd-pam) だった。
- 日次timerでも同じ確率で失敗し、journal に毎日 failure が残り得る。#1990 の受入（日次 cycle の成功）に関わるので、この issue の範囲で直す。

## 求める動作
- 読めなかったプロセスについて、同じ個体（pid と starttime が同じ）を短い間隔で数回だけ再確認する。消えた・別個体になった場合は exited として扱い、読めるようになった場合はその時点で完全に検査し（fd・cwd・root・maps・mount参照）、使用中なら busy とする。回数と待ち時間の上限内で証明できないまま存続する場合は、従来どおり unknown として止める。**読めないプロセスを使用していないものとみなす、静かなフォールバックは入れない。**
- 再確認の回数・待ち時間・結果（再確認で解消した pid の数など）を、出力の `process_scope` などに記録し、後から確かめられるようにする。
- まず原因の仮説を一時環境で検証する。exec 中の短命プロセスで fd だけが EACCES になる状況を再現できるか確かめる。再現できなければ、そう報告したうえで、上記の再確認方式が仮説に依存せず安全である理由を書く。
- 既存の安全契約（nondumpable の証明、CapPrm の比較、mount 参照、hydrate/evict の中断境界）を弱めない。

## テスト
- 修正前に失敗する再現テスト: 1回目は fd が EACCES で maps が読め、2回目には (a) 消えている、(b) 読めて無関係、(c) 読めて対象を開いている、(d) 読めないまま存続、の各場合。期待はそれぞれ exited、通過、busy、unknown。
- 既存の usage・crash/recovery 境界のテストが通ること。

## 完了条件
- 新PR、CI の Python 成功、README（3言語）の該当箇所の更新（使用中判定の記述があれば）。
- 実機の読み取り専用の確認: worktree のコードから `--repo main` で scan と `cycle --dry-run` を複数回（例: 10回）実行し、unknown と failed の回数を報告する。修正前後で比べられるとよい。実機の配置・timer・sudo には触れない。
- 最終報告: 原因検証の結果、再現テスト、修正、テスト件数、CI の URL と最長 shard の時間、実機での回数。作業時間の上限は2時間。
