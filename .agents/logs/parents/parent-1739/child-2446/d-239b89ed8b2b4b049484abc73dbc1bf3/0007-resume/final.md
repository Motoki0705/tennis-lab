`main` をマージして統合PR #2470をpushし、Draftを解除しました。本文の先頭に #2444・#2445・#2459・#2446 の `Closes` を記載し、GitHubで競合が解消されたことを確認しました。 [PR #2470](https://github.com/Motoki0705/tennis-lab/pull/2470)

- 採番、設定登録、ソース識別、FP32回帰テストの4件を解消しました。履歴fixtureを使った回帰テストに加え、キャッシュ識別が16条件すべてで取り込み元の `main` と一致することを確認しました。
- CPUテストは1,742件成功、警告1件、366.27秒。ECPose・FastHMR・単眼PLCSと新経路を検証し、pre-commitとknowledge検査も通過しました。
- 計測済みの新経路では69/69本で選手と球を確認できました。Meiji 3cam未見3本のIDF1は旧92.07%・新80.42%、選別recallは旧85.30%・新67.25%。4camの未使用の対応視点への再投影残差は旧7.89 px・新24.86 pxです。 [計測記録 #2493](https://github.com/Motoki0705/tennis-lab/pull/2493)

ブランチは `codex/parent-1739-issue-2446-primary`、HEADは `b62a0f3ddca17145394f87992a3b03012051466f` です。 [CI実行中](https://github.com/Motoki0705/tennis-lab/actions/runs/45768941192)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2470,"head":"b62a0f3ddca17145394f87992a3b03012051466f","resume_on_success":false}}