# #1913 調査の途中成果（WIP・mergeしない）

2026-11-12、作業環境をラップトップからデスクトップへ移すため、調査worker（Codex planning）を途中で停止した。
`report.md` は未作成。このbranchには再取得しにくい小さな解析結果と指示書・照合scriptだけを置く。

- `task.md`: 調査workerへの指示書（再委任時にそのまま使える）
- `*.py`: workerが書いた照合・再計算script
- `analysis/`: 再計算結果（Court 9評価、#1859 benchmark、snapshot hash照合、PR一覧）。1MB超の `capacity-metrics.json` / `query-control-metrics.json` は除外した（Driveから再計算可能）
- `evidence/index.json`: 取得したDrive一次JSONの対応表（path, drive_id, size, sha256）。一次JSON本体（約31MB）とAPI応答キャッシュ（約37MB）はpushしていない。index.jsonのDrive IDから再取得する

未着手・未完了: 項目7（#1231探索）の一部以降、項目8（元issueのクローズ可否）、項目9（sequence競合）と report.md の作成。
