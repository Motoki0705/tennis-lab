SceneResult v3と `event_detection` の3方式を実装し、[PR #2610](https://github.com/Motoki0705/tennis-lab/pull/2610) を作成しました。ブランチは `codex/parent-1722-issue-2605-primary`、HEADは `59a62db3cd7593607c9bde0bfcc3e1c3f2c67d2e` です。

同じHEADで `multiview-v3` の全69クリップを生成しました。元のv2配列は全クリップでビット一致し、16,302ファイルをhard linkで共有しています。再実行で913イベント、判定不能1,885フレーム、47/11/11の分割、全ファイルのSHA-256を照合しました。[manifest.json](https://github.com/Motoki0705/tennis-lab/blob/59a62db3cd7593607c9bde0bfcc3e1c3f2c67d2e/data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3/manifest.json)

関連テストは1,811件が共有venvで通過し、`os.pidfd_open` がなく停止した2件もPython 3.11.17での再実行で通過しました。pre-commitのRuff・mypyは通過し、[GitHub CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45763214951) は実行中です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2610,"head":"59a62db3cd7593607c9bde0bfcc3e1c3f2c67d2e","resume_on_success":false}}