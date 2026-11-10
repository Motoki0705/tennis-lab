[PR #1811](https://github.com/Motoki0705/tennis-lab/pull/1811) の `pr-1811-1` を修正しました。同branchへ通常commit `060508f1b` を追加済みです。

- インラインTeXと一般命令を拒否し、除外範囲・通貨の規則をREADME 3言語へ明記。
- 関連102テスト成功（24.82秒）。追加ケース0、pre-commit・全CI成功。
- 公開済み4件のstaged bytesも新検査に成功し、HTML不変を確認。実機再生成は0回です。

[再検査記録](https://github.com/Motoki0705/tennis-lab/blob/060508f1bdabc189c291b83ab8544b5b5731b80d/outputs/explain-html/robustness/tex-review/staged-validation.json)を保存しました。指摘解消の確認・mergeは親へ返します。