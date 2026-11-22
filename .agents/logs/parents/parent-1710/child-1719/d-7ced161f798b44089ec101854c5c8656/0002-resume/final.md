[PR #1793](https://github.com/Motoki0705/tennis-lab/pull/1793)をmainへ載せ替えて更新しました。

- **HEAD:** `9c02a7cbcafa73f5dc6dd88bfd521e4143e4fce7`
- **[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42587926285):** Python・Lint・CPU全10ジョブ成功
- **載せ替え:** 旧境界`9b4ec527a`から、このPRの3コミットだけをmain `f8b3f244c`へ適用。README 3言語版はmainのvisual説明を保持して重複を解消しました。接続変更を加え、計4コミット・11ファイルです。
- **接続:** 実`render_document`・`validate_visuals`をservice既定経路へ接続。GeminiがFigure型で示す新規画像要求は、実`UnavailableImageGenerator`経由で拒否し、現行版を保持します。
- **検証:** 追加integration 1件で図編集のcommitと画像要求拒否を確認。関連**193 passed／23.63秒**、追加caseはsetup込み2.20秒。恒等compilerへの故障注入も検出し、pre-commit・mypyも成功。実API呼出しは0回です。
- **公開面:** HTTP入口・状態名・token/fragment受渡し・JobController signatureは変更なし。CLIの`--compiler`は任意の追加変換になりました。

作業treeはclean、PRはmerge可能です。実browser品質validatorとUI実演は引き続き#1720/#1721の残作業で、レビュー・merge・merge後main CIは親へ引き継ぎます。