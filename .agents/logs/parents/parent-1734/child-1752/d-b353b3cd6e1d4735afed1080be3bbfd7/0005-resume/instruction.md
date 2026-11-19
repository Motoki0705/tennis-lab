# PR #2227: mainへの付け替えと最新mainの取り込み（親 #1734 から #1752 へ）

土台のPR #2182（#1751）がmainにmergeされました。PR #2227 のbaseを `main` に付け替えました。mainには、ほかに #2244（共有court sideの検証）と #2240（人物ステージの精度選択とDINO拡張のpin）も入っています。

1. `origin/main` を `codex/video-batch-cli` にmergeし、衝突を解く。両方の意味を保つこと。#2182の最終版・#2240・#2244の変更を消さない。このPRの一括処理CLIの変更も消さない。
2. 統合後に、次のことを確かめる。
   - 一括処理CLIが、#2240の精度設定（ViTPose・DINOのfp16指定）を受け取り、正しく渡すこと。
   - 一括処理CLIが、#2244の共有court sideの検証を通ること。
   - 69clipの本番コマンド（`knowledge/runs/run-i1752-shared-fp16-20261118/full69-command.txt`）が、統合後のコードでそのまま有効であること。CPUでdry-runやplanの確認ができるなら行う。
   - 必要ならコマンドファイルを更新する。
3. `tests/unit/tennis_scene` 全体と、関連するCPUテスト、pre-commitを実行する。GPUは使わない。本番の実行もしない。
4. pushしてPR CIを起動し、対象HEADで `Python` を含む全ジョブが成功するまで見届ける。
5. PR本文を最新の状態に合わせて更新する。Draftのままでよい。

CUDAのエラーやハングが起きたら、再試行せずに止めて報告すること。最後に、HEADのSHA・CIのURL・衝突の解き方・本番コマンドの状態を簡潔に報告すること。
