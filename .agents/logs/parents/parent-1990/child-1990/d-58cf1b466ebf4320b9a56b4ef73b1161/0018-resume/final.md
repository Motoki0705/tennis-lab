[PR #2148](https://github.com/Motoki0705/tennis-lab/pull/2148) を更新しました。branchは `codex/storage-tiering`、HEADは `83fc9c6c5`。push済み・worktree clean、`Refs #1990` を維持しています。

- **pr-2148-1:** PathResolverが同じconfigの宣言リンクだけを許可。Clip Studioのraw/processed発見・project保存復元と、合成pipelineの入力・workspace解決を通し、許可／宣言外／不一致／未mountを検証しました。
- **pr-2148-2:** dataset container直下の版rootも、リンクを辿らず検査して拒否します。
- **pr-2148-3:** 無関係な明示単位のdry-runではリンク先を読みません。宣言リンク自身のatimeだけを例外とする契約を3言語に記載し、payload・他metadata・状態・directoryの不変を検証しました。

追加29件。実装前に9件の失敗を確認し、最終回帰セットも旧コードで13件失敗しました。consumer関連497件、監査・文書132件、pre-commitが成功。全6,948件の実行では6,946件成功・旧mount fixtureの不一致2件があり、補正後に該当を含む実mount全10件が成功しました。

**対象HEADの[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43732365470)はPythonを含む全14ジョブ成功。** 最長shardは37分6秒。同じ実装の先行HEADは25分46秒で、時間差の原因は未調査です。

実機scan・cycle dry-runは両方unknown 0。root symlink・E: smplxのmetadata、状態directoryの有無は不変でした。配置未切替の3エラーは残っています。

[更新手順書](https://github.com/Motoki0705/tennis-lab/blob/83fc9c6c5b1af507aa3780591853faa91af0e8cb/tmp/storage-tier-phase2-runbook.md)・[実行記録](https://github.com/Motoki0705/tennis-lab/blob/83fc9c6c5b1af507aa3780591853faa91af0e8cb/tmp/storage-2148-worker-record.json)を保存済みです。実機の配置切替・index・hydrate・削除・unit・sudoは未実施。残りは親のレビュー・mergeと移行指示です。