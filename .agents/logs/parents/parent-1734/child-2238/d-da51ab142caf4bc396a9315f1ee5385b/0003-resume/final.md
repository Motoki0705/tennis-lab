再レビューの指摘を `codex/person-stage-speed` の `065cb3fb1` で修正しました。PRはReady for reviewです。 [PR #2240](https://github.com/Motoki0705/tennis-lab/pull/2240)

公開のみを再開するときは、保存receiptに記録したDINO・ViTPoseの設定精度を照合し、片方でも異なれば公開前に停止します。精度欄のない旧receiptは、両モデルともFP32の場合に再開できます。一致する混合精度での再開も、既存の統合テストで確認しました。

通常のキャッシュ再利用とload専用実行はコンポーネントidentity、datasetの完成判定は公開identityで精度を照合します。基準mainが保存した31段の成果物も、既定・明示FP32の両方で引き続き再利用できました。

関連CPUテスト1,561件、pre-commit、対象HEADのCI全14ジョブ（`Python` を含む）、Knowledge libraryが通過しました。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44291403517)・[Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44291404137)