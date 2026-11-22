`pr-2043-1` を修正し、同じbranch `codex/citation-map-webui` の [PR #2043](https://github.com/Motoki0705/tennis-lab/pull/2043) へpushしました。HEADは `6d357c7be` です。

- 不正文書のファイル名・理由を表示し、影響する探索だけを停止します。外部ID衝突は全参加文書を無効化し、帰属不能の場合は全体エラーを維持します。Pythonの拒否条件は変更していません。
- lint・unit 210件・build・e2e 11件・関連pytest 369件が成功しました。
- 対象HEADの [Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/43683282516)・[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43683282515) は全ジョブ成功しました。
- PRと子issueへ証拠・画像を記録し、`Closes #2024` を維持しています。

親による指摘解消の確認・merge・merge後main確認が残ります。