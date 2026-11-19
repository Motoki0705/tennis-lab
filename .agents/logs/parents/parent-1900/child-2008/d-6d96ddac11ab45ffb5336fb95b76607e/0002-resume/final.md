[PR #2018](https://github.com/Motoki0705/tennis-lab/pull/2018) を修正し、`codex/worker-session-replacement` へ通常pushしました。

最終HEAD: `b44d08abc04bc9369d546b3a38c090331d5fa947`

取消履歴だけになった旧委任でも、保存した付替え元runを再照合して再試行できます。取消ターンによる成果照合の拒否と、旧記録の保持は維持しました。

追加テストは次の2件です。

- `test_legacy_session_replacement_retries_rejected_launch_and_requests_review`
- `test_legacy_session_replacement_retry_rejects_changed_source`

修正を外すと、前者が `no non-cancelled worker turn to verify` で失敗しました。復元後は関連81件・parent関連454件成功、pre-commit成功。[最新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43589518158) も全ジョブ成功です。検証processは終了・回収済みです。

#1961はpreflight version 2、非取消2ターン・取消1ターンで、今回の組合せには該当しません。読み取りだけで確認し、対象25ファイルとtaskのSHA-256が不変でした。

残作業は親による統合レビュー・mergeと、merge後の#1961への実適用です。