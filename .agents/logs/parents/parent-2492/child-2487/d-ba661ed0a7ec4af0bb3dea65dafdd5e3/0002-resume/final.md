[PR #2499](https://github.com/Motoki0705/tennis-lab/pull/2499) に更新済みの土台をpushし、[新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45393959250) が起動しました。`codex/parent-2492-issue-2487-primary` に #2498 の `b1011e406` と #2497 の `d64962a28` が含まれることを引き渡し前に確認しました。

旧CIで失敗したFP32 cache互換2件とforward境界検査1件を含む **35件がローカルで成功**しました。6,844画像の再採点でも、生予測と採点JSONの計4ファイルが既報の出力とバイト一致しました。補助担当向けの計測API記載も `measure_forward()` に更新済みです。[#2487 のadapter契約](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7129890902)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"156502ddebad5699de5ef59d1f6a2327e384319c","resume_on_success":true,"next_step":"#2491の共通recipe実装と補助担当のECPose adapterを照合して経路接続・既定recipe回帰検証を進め、公開済みの#2496速度基準と候補成果を同じPRへ統合する。"}}