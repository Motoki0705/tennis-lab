このターンは調査と設計の報告だけを行う（コードの変更・commit・PRは作らない）。担当issue #1961 の Acceptance checklist を満たす実装方針を決めるための材料を集め、親が判断すべき点を明示して報告する。作業時間の上限は60分。以下の5点すべてに答えたら終了してよい。答えられない点は、試したことと不足している情報を書いて終了してよい。

1. merge: `.agents/ops/worker_logs/publication.py` の `publish()` で `mergeable` が null のとき止まっている箇所（2か所）を、HEAD指定のmerge API要求へ置き換える設計。GitHubが mergeable 未計算のときに merge API が返す応答（成功・405・409など）の公式ドキュメント上の扱いと、拒否を記録する状態名。既存の安全検査のうち mergeable に依存するものがあるか。
2. 続行の契機: 同じ親の「次の書込みを伴うCLI操作」として、保留中の公開を続ける入口の具体的な一覧（run-worker・publish-worker-log・release・handoff・自律側の start-worker / continue-worker / workflow-report など）と、各入口で公開を続けるタイミング（操作の前か後か、失敗したら操作自体をどうするか）。親の操作ロックやregistryロックとの関係。status / worker-log-status を読み取り専用に保てることの確認。
3. 担当終了・引継ぎ時の照合: `release --ended` と `handoff` で、workerログの保留と未公開の判断ログ（decision_logs の publication）をどう照合・続行し、残った場合に何を出力・記録するか。終了自体は拒否しない。
4. ランナー判定: `.agents/ops/lib/agent_runtime.py` の `events.failed` と、#1877 turn 3 の事例（`Reconnecting... 2/5 (workspace routing discovery timed out)` の後に turn.completed と最終応答あり）。Codex CLIのイベント列で「回復した一時エラー」と「致命的エラー」をどう区別できるか（イベント種別・順序・message の形式）。誤って真の失敗を成功にしない判定条件の案。`.agents/ops/worker_logs/sources.py` の公開判定への影響。
5. 既存の取り残し: このホストのparent state（`${XDG_STATE_HOME:-[local path]}[local path]` の `worker-logs/`）を読み取り専用で走査し、journal の state/stage 別の件数と、2026-11-13時点でopenのworkerログPR（`gh pr list --state open --search "worker-log in:head"`、22本）との対応表を作る。担当中でない親（#1723 など）の journal を含め、#1900 の親が一括で続ける入口の設計（既存の `publish-worker-log` の拡張か新しい入口か、所有者検査をどう扱うか、dry-run の出力）。`blocked / absolute_path` と `captured / awaiting_final` があれば件数と原因の見立ても添える（修正は対象外）。

stateやjournalは読むだけで、書き換え・公開・PRのmerge・closeはしない。報告の最後に、親に判断してほしい点を箇条書きでまとめる。
