同じbranch `codex/citation-explore-cli` で修正し、[PR #2044](https://github.com/Motoki0705/tennis-lab/pull/2044)へpushしました。HEADは `65f9fffc7`、作業ツリーはcleanです。

- OpenAlexの速度制御・再試行・照会待ち保存、引用元の候補照合、OA取得候補の記録を実装しました。
- WASB再実行は135.681秒。既読5本を再利用し、追加agy呼出は0回。59行を新たに同定し、未同定は133→38行、照会待ち36行、取得失敗2本、判定不能38行です。結果文書もコミット済みです。
- 関連147件、既存互換を含む1062件、kg_validate、pre-commitが成功。最終HEADの[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43610771347)・[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43610771339)も成功しました。

深さ2探索は**未完了**です。OpenAlexのRetry-Afterが33838秒で6時間上限を超えるため、中断状態で保存しました。再試行指定は **11/15 09:00:58 JST以降**。残り36行の照会は、サービス利用可能後に1〜3分、追加読解0本の見込みです。

保存形式へ追加した`pending`・`lookup`のWeb UI側対応確認と、再開後の完了確認・統合レビュー・mergeが残ります。validatorは未指定のため試行0／完了0です。