本生成を止めていた `rally_000043` を同じseedで再現しました。ネット支柱の外を通る候補が着地点のセル判定まで進んでいたため、ネット接触を確認できない候補をセル判定前に再抽選するよう修正しました。

`codex/parent-1722-issue-2567-primary` の `e4438667dd41162c6f4972a92146eb9b59bbba54` にコミットしました。生成の統合テスト2件、BLCSシミュレータの単体テスト44件、pre-commitが通過しました。

初回の未完成データを退避し、`data/ball_refiner/single_object_net_v1` の1280ラリー生成を同じ設定で再起動しました。生成完了後にmanifestを照合し、学習登録とPR作成に進みます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261125T175618Z_b6062ef3c7b9_ball-refiner-net-v1-generate-retry"]}}