前回のturnは、親側のCLIプロセスがメモリ不足で回収された影響で 08:45 JST 頃に途中で止まり、その後 09:20 頃に WSL が再起動した。プロセスはすべて消滅し、親が終了を記録済み。

止まった時点では、commit f7f402f46 を push し PR #2085 を作成済みで、最終確認の pytest（test_git_sync・test_interactive_checkout・test_checkout_job・test_ops_control）を実行中だった。worktree に未commit差分はない。

状態を確認して、残りの確認（関連テスト、pre-commit、修正を外すと新テストが失敗することの確認、PR本文、対象HEADのCI）を最後まで進めてほしい。制約は最初の指示と同じ（pytest -n 4 以下、systemd unit 操作・git-sync のONをしない、メインcheckoutを変更しない、force pushしない）。
