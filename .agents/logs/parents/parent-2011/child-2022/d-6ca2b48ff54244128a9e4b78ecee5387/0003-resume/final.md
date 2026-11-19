major 2件を修正し、同じbranch `codex/citation-explore-cli` の[PR #2044](https://github.com/Motoki0705/tennis-lab/pull/2044)へpushしました。HEADは `68e02f1c1`、作業ツリーはcleanです。

- 本文候補を対象ID・metadata・題名で確認し、参考文献や補足PDFの混入を防ぎました。検索は全cursor・件数を確認し、網羅性未確認は`unverified`で保存します。
- 既読5本の取得元は正常でした。旧検索のWASB 77行・MonoTrack 1行を未確認へ戻し、分類と読解を保持しました。現在は同定済み103、未同定38、照会待ち36、一意性未確認78行。取得失敗2本、判定不能38行です。
- 関連171件（今回追加24件）、kg_validate、pre-commit成功。[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43617368554)・[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43617368595)も成功しました。追加validatorは未指定、試行0／完了0です。

CI確認完了は **11/15 02:10 JST**。指定の09:00:58より前なので、WASB探索は再開していません。残件は114行の再照会と深さ2の完了、Web UI側の`unverified`対応確認です。修正内容・更新文書・検証証拠はPR本文に記載しました。