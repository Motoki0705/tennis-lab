PR #2610の正式レビュー3件を修正し、既存branch `codex/parent-1722-issue-2605-primary` に `cba442a2d9aeb21d413133aa421770e627a8755f` をpushしました。 [PR #2610](https://github.com/Motoki0705/tennis-lab/pull/2610)

- `cc35f62a5` は自動導入と3言語の手順で、事前診断にサンプル再構成と同じ方式を渡します。資産が揃った状態を模したCLIテストで終了値0を確認しました。
- `63b4824a9` はTensorRT書き出しと途中段階の処理をイベント設定から切り離しました。書き出しCLIがモデル起動直前まで進むことを確認しました。
- `cba442a2d` は注釈を固定v2 manifestの映像ハッシュに束縛します。実クリップ `clip_040` で8件を採用し、同じフレーム軸でcam0のハッシュだけを差し替えると拒否しました。

修正前に回帰4ケースの失敗を再現し、修正後は関連185件が通過しました。`multiview-v3` を同じHEADで再生成して配布先に反映しました。`artifact_id` と `execution_key` は各69件で変わりました。前回版との照合では、全69クリップのイベント配列・coverage計552個がバイト一致し、採用シーンNPZのSHA-256も全件一致しました。前回版は #2600 の再現用に `.multiview-v3-555a246aa` に保管しました。

新しい配布manifestのSHA-256は `cdc76e6b6be2dbdcb0c8992a2e25e94a19b3b404572dc87e72cdb57e7479633b` です。PR本文の実行証拠を更新し、修正HEADのPython CIを親CLIへ引き渡します。 [配布manifest](https://github.com/Motoki0705/tennis-lab/blob/cba442a2d9aeb21d413133aa421770e627a8755f/data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3/manifest.json)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2610,"head":"cba442a2d9aeb21d413133aa421770e627a8755f","resume_on_success":false}}