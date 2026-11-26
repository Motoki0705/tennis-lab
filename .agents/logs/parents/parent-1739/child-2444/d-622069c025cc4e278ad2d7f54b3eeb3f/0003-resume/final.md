3D骨格CLIを実行入口の一覧に登録し、成果物を読む前に入出力パスを検証するよう修正しました。同じブランチ `codex/parent-1739-issue-2444-primary` に新HEAD `ecd5843d11a1b88bfdc993b3e51862c22f273145` をpushしました。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449)

CIで失敗した2件と新規の異常パステスト2件を含むpytest 493件、ruff、mypy、pre-commitが通過しました。Meiji 4camの `clip_012` を通常CLIで再実行し、1,017フレーム・7,116候補の出力配列20点のSHA-256が修正前と一致しました。CLIの変更に伴い成果物IDは更新されました。

#2442の確定疑似正解によるダブルス採点は残件です。新HEADのCIは実行中です。 [CI run #45232921220](https://github.com/Motoki0705/tennis-lab/actions/runs/45232921220)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2449,"head":"ecd5843d11a1b88bfdc993b3e51862c22f273145","resume_on_success":false}}