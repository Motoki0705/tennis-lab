このターンは、壊れたsessionからの正規の付替えによる新しいsessionです。前の2つのsessionの経緯は次のとおりです。
- 実装ターンのsessionは、2026-11-13 のWSL停止で失われた。
- 親が起動した引継ぎsessionが未commitの修正を仕上げ、da46fdda81e7f0d3103e13dd4fa8a2ae556af0c5 としてpushした。PR #1984 はdraftを外し済みで、そのHEADのCIは全ジョブ成功。実装の内容はPR本文にまとまっている。

このターンでやること:
1. PR #1984 は現在、mainとの競合で mergeable でない。origin/main を branch へ merge して競合を解消する（rebase・force push はしない）。mainには、子 #2008 の `run-worker --replace-session`（parent/manual_workers.py、worker_preflight.py、cli.py など）と、その他の親CLIの変更が入っている。どちらの意図も失わないように解消する。
2. 解消後、PR #1984 の変更に関係するテスト（worker_logs・decision_logs の公開、parent の続行・終了・引継ぎ、lock、dispatcher supervisor、agent_runtime、worker log recovery）と tests/e2e/agents_ops の parent 関連を実行し、pre-commit を通して通常のpushをする。
3. 最新HEADのCIを確認し、PR本文の検証欄を更新する。
- pytest の並列は `-n 4` 以下。一時ファイルは [local path] ではなく worktree 外の state 配下に置く。起動したpytestなどのプロセスは、終了前に必ず止める（前のsessionは終了時に子孫プロセスが回収された）。
- 作業時間の上限は3時間。止めてよいのは、方針と矛盾する事実が見つかった場合と、3時間を超えた場合だけ。
- 最終報告には、最終HEAD、競合の内容と解消方法、テストとCIの結果（失敗があれば基準commitでも同じかどうか）、未解決の点を書く。
