規則検出CLIのDATA・OUTPUTパス契約を登録し、出力先がsymlinkで指定rootの外に出る場合の回帰テストを追加しました。main側のFastHMR importエラーには、既存の修正を取り込みました。 [FastHMR修正 PR #2573](https://github.com/Motoki0705/tennis-lab/pull/2573)

関連テスト44件、リポジトリの設定監査、静的import検査、pre-commitが通りました。修正HEADで `clip_012` を再実行し、23候補すべてが全69件実行時の結果と一致しました。契約台帳全体の照合を含むPython CIは新HEADで確認します。

`codex/parent-1722-issue-2568-primary` の `210583344b3dde5c4c4c5e11377e26ffdd9ed341` を既存PRにpushしました。合成評価の記録PRはKnowledge CIに成功しています。 [実装 PR #2569](https://github.com/Motoki0705/tennis-lab/pull/2569) · [記録 PR #2570](https://github.com/Motoki0705/tennis-lab/pull/2570)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2569,"head":"210583344b3dde5c4c4c5e11377e26ffdd9ed341","resume_on_success":false}}