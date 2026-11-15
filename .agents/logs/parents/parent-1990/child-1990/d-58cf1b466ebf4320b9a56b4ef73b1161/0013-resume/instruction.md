# PR #2023 5回目レビュー（parent-1990-73b724af5ebc494d8ca8ddba5e0988c8, HEAD 3e5fdbeaa）の指摘修正

前回の修正は親が確認済み（ローカル188件成功、新テスト13件が修正前で失敗、CI成功）。5回目のレビューは、agentの出力後にGitHub APIのrate limit（HTTP 403）で事後照合が失敗し、正式結果は `incomplete` になった。保存された出力にmajor 1件があるので、これに対応する。静的推論なので、**先に再現テストを書いて失敗を確かめてから**直すこと。今回も配置切替・実削除・unit操作・sudoの操作は行わない。

## pr-2023-1（checkpoint_files.py:103 付近）保存済み証拠の不一致を拒否するようにした結果、中断した公開の復旧が完了できない
指摘（原文の要約）: 既存のmanifestがある状態で、mirrorのrename交換の後、recordの前に停止すると、E:は新しいinode、manifestは古いinodeのまま残る。README.ja.md:56–58の手順どおりに集合を照合してjournalを退避しても、前回追加した `check_recorded` がindexを拒否する。verify・mirror・hydrateも、同じ古い証拠との比較で止まる。古いpayloadを戻しても、renameによるctimeの変更が残る。recoverにも、この投影を修復する経路がない。文書化した復旧を完了できない。
- 求める動作: 通常の不一致の拒否（前回の修正）は維持する。そのうえで、中断した公開について、古い証拠と履歴を保存したまま、明示的に復旧を完了できる経路を用意する。たとえば、recoverがjournalに記録した交換の意図と、E:の実体（新しいinodeの内容が、D:の照合済みhashと一致すること）を照合して、新しい証拠を記録し、古い証拠を履歴へ退避する。未照合の内容を正としない。黙って再indexで上書きしない。
- 通常の単位（ディレクトリ）の公開でも、同じ中断点で同じ問題がないかを確認し、あれば同様に直す。
- テスト: `test_interrupted_root_file_publication_keeps_old_inode_and_blocks_retry` を、復旧の完了まで広げる（復旧後にverify・mirror・hydrateが通ること、古い証拠が履歴に残ること、未照合の内容では復旧できないこと）。README（3言語）の復旧手順を、実装と一致させる。

## 注意
- GitHub APIのrate limitは 2026-11-15 10:57 JST ごろにresetされる見込み（多数のsessionが再起動直後に消費した）。push・PR更新・CI確認でHTTP 403が出たら、resetを待って再試行すること。待機のために作業を終了しない。

## 完了条件
- 再現テスト（修正前に失敗すること）、修正、README（3言語）、PR本文（`Closes #1990` を維持）。
- 対象HEADのCIで `Python` 成功まで確認する。
- 最終報告: 再現と修正、テスト件数。作業時間の上限は1.5時間。
