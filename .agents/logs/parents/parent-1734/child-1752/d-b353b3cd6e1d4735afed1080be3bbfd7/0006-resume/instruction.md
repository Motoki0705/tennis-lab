# PR #2227: mainの取り込みと正式レビューの準備（#1752）

先行の#2225と#2226に続いて、#2270（court_sideのplayer証拠）もmainにmergeされました。

- `origin/main` をmergeし、衝突を、両方の意味を保って解く。
- 2026-11-19の範囲の変更に合わせて、一括CLIの本番の構成を明示的に選べるようにする。
  - 本番の構成: ball検出を外し、3Dはプレーヤーだけにする。court_sideは `evidence=player` で判定する。
  - ball推論ありの構成: #1757のbenchmarkで使う。
  - #2270の設定を使い、同じ判定を別の場所に作らない。
  - 69clipの本番コマンドの記録（`knowledge/runs/run-i1752-shared-fp16-20261118/full69-command.txt`）は、新しい構成に合わせて更新する。69clipの実行はしない。
- **GPUは使わない。** 今、GPUの停止の原因を切り分けている最中なので、queueにjobを入れない。CPUのテストと保存済みの成果物の再生で確かめる。
- 最後に、PRをReadyにして、HEAD・CI・CPUでの確認の結果を報告する。正式レビューは親が依頼する。
