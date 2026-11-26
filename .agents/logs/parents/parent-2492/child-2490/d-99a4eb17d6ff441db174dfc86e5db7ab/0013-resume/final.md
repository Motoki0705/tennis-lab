[PR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500) のレビュー指摘3件を修正し、既存branchに `1bce3e5972a46f1f8ade71513aa9d582995558d2` をpushしました。

1. **図表のbatch表記**　速度・VRAMのbatchと精度のbatchを明記しました。精度はbatch 8の予測を使用し、batch 1・4は未計測です。再生成した30行の既存値と27点のPareto判定は旧HEADと一致しました。
2. **比較CLIの入力検証**　固定インデックスのSHA-256と、欠測条件に対応する終端ログを照合します。実レポートのP50を1 msに差し替えた場合と、存在しないログを指定した場合は、出力前に拒否されました。
3. **精度ログの保持**　RTMPoseの精度ログ6件を保持先に保存し、共通・専用の読込を揃えました。保持ログ31件のSHA-256を照合し、元ログを参照できない環境で精度18条件の読込を確認しました。

回帰テスト7件を追加し、関連pytest 113件、pre-commit、knowledge検査が通過しました。[新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45638041515) と [Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/45638041116) は結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"1bce3e5972a46f1f8ade71513aa9d582995558d2","resume_on_success":false}}