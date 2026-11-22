# 中断した前回ターンの続き（PR #2023, fix-1990-9 の指示）

前回のターンは、メモリ不足でプロセスが停止したあと、WSLのshutdown（2026-11-15 09:20頃）で中断した。あなたのプロセスはその時点で終了している。[local path] は再マウント済み。

親が確認した中断時点の状態:
- branch `codex/storage-tiering` は push済み。HEAD `f17898523`（`c87d264c8` Split checkpoint mirrors and pin runner assets to E storage、`f17898523` Preserve tier-specific checkpoint container permissions）。worktreeはcleanで、未commitの変更はない。
- 対象HEAD `f17898523` のCIは、`Python` を含む全ジョブ成功。
- 最終報告はまだない。

前回の指示（fix-1990-9）の完了条件のうち、残っているものを仕上げること。まず、中断前にやり終えた範囲を自分で確かめる（`tmp/` 配下の途中成果を含む。再起動で失われたものはやり直す）。
- 指示の1〜3が実装・テスト済みか。テストが修正前のコードで失敗することを確かめたか。
- 4: 実機の読取専用QA（ckptのサブ単位を含めて、unknown 0、atime等が不変）。
- README（3言語）、PR本文（`Closes #1990` を維持）。
- 最終報告: 変更点、テスト件数、実機QAの要点、第2段の手順案の更新部分（sudoが要る手順の一覧。どのコマンドを、いつ実行するか、学習・CIへの影響。現在のbindは既にE:を指しているので、切替前の再インストールは不要という点の確認を含む）。

追加の変更でHEADが変わったら、対象HEADのCIで `Python` 成功まで確認する。今回も配置切替・実削除・unit操作・sudoの操作は行わない。作業時間の上限は1.5時間。
