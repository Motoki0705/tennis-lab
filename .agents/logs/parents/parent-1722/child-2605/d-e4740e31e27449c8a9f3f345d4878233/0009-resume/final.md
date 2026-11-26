PR #2610の正式レビューで採用された3件を、`codex/parent-1722-issue-2605-primary` に個別コミットしました。 [PR #2610](https://github.com/Motoki0705/tennis-lab/pull/2610)

- `cc35f62a5` は自動導入と3言語の手動手順で、サンプル再構成と同じイベント方式を事前診断に渡します。資産が揃った状態を模したCLIテストで終了値0を確認しました。
- `63b4824a9` はTensorRT書き出しと途中段階の検証をイベント設定から切り離しました。書き出しCLIがモデル起動直前まで進み、再構成と配布では方式指定が必須のままであることを確認しました。
- `cba442a2d` は注釈を固定v2 manifestの全カメラのSHA-256に結び付けます。実データの `clip_040` で8件を採用し、同じフレーム軸でcam0のハッシュだけを差し替えた入力を拒否しました。

修正前は回帰4ケースが失敗し、修正後は該当ケースが通過しました。最終HEAD `cba442a2d9aeb21d413133aa421770e627a8755f` で関連回帰と69クリップのCPU再生成を開始しました。前回配布版 `40efd634…` とのイベント配列・coverageのバイト照合が済み次第、配布先とPRの実行証拠を更新してpushします。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T081104Z_a478ba7f9e17_scene-events-v3-review-major"]}}