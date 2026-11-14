# #1990 第1段の追加修正（PR #2023）: 能力集合の差で読めない同UIDプロセス

残った systemd（PID 544、`systemd --user`）を親が実機で確認した。`[local path]` は `CapPrm: 0000000800000000`（CAP_WAKE_ALARM、user@.serviceのambient由来）で、検査側の `CapPrm` は0。カーネルのptrace read検査（commoncapの `cap_ptrace_access_check`）は、対象の permitted 集合が呼出側の permitted 集合の部分集合でない場合に拒否する。実際に `[local path]` と `cwd` は EACCES、`fd` の一覧は読めるがリンク先は読めない。

これは非dumpableと同じく「同UIDでもカーネルが設計上読取を拒否する」場合なので、**親の判断（合意の範囲内）として同じ除外規則に含める**。新しい人間合意は不要。

- 除外の根拠に「対象の `CapPrm` が検査プロセス自身の `CapPrm` の部分集合でない（同じuser namespace内）」を追加する。両方の値を `[local path]` と `[local path]` から読み、判定に使った値を監査記録・ログに残す（件数・comm・根拠の種類）。プロセス名による許可リストは作らない。
- 非dumpableでも能力集合の差でもないのに読めない同UIDプロセスは、従来どおり失敗にする。
- テスト: 擬似procで、CapPrmが部分集合でない同UIDプロセスが除外・記録されること、部分集合なのに読めない場合は失敗すること。READMEの前提（3言語）に追記する。
- 実機の読取専用dry-runで、プロセス検査による失敗が無くなり、残る失敗が配置切替前であることだけになるのを確認する。
- PR #2023 を更新し、対象HEADのCIで `Python` 成功まで確認する。PRはdraftのままでよい。

作業時間の上限は1.5時間。
