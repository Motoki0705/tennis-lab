[PR #1818](https://github.com/Motoki0705/tennis-lab/pull/1818) はmainへ追従・force-with-leaseでpushし、baseをmainへ変更済みです。branchは `codex/explain-html-claim-faithfulness`、HEADは `f15c2b9e0`。

107 tests passed（34.43秒）、pre-commit・ruff/mypy・[対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42661709046)は成功しました。

追加実機は直列11/12回、各題材2試行で終了。最終3候補は独立検査を通りましたが、worker照合では以下が残り、**受入未達のdraftを維持**しています。

- 三角測量：非同期時の誤差を過度に断定。
- 配置比較：視野減少・配線に関する条件不足。
- 観測誤差：固定視差での基線長と感度の説明が掲載式と逆。

観測誤差は2回とも正常終了しました。一方、三角測量1回目で候補出力後の900秒timeoutを記録しました。

[評価表・反例・終了不良の証拠](https://github.com/Motoki0705/tennis-lab/blob/f15c2b9e0bfcd916f125dbdce69a413be5056e0e/outputs/explain-html/faithfulness-followup/evaluation.md)を保存し、子issueとPRへ要約済みです。人間評価欄は空欄。残る見逃しとCLI終了不良の扱いを親判断へ返します。merge・closeは未実施です。