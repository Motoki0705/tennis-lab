PR #2610のCIで失敗した設定監査に対応し、ベンチマークCLIを監査台帳に登録しました。v2と注釈の入力ディレクトリを、生成前に検証します。ベンチマークの単体テスト4件とpre-commitのRuff・mypyが通過しました。 [PR #2610](https://github.com/Motoki0705/tennis-lab/pull/2610)

修正は `codex/parent-1722-issue-2605-primary` の `555a246aabb7201959e3bb28cc6f40079e62b5c3` にコミットしました。CIで失敗した2件を含む監査の再実行を親CLIに引き渡します。監査通過後に同じPRへpushし、CI成功後に修正HEADで69クリップを再生成・照合します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T062127Z_83d0359d882e_scene-events-v3-config-audit"]}}