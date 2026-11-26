正式レビューの2件を修正し、`codex/parent-1722-issue-2567-primary` の `4afdccc7c348a946cc5bb86aac583ae220ce652f` をpushしました。FastHMR修正 #2573 を含むmainを取り込み、関連テスト20件、静的import検査、pre-commitが通過しました。 [実装PR #2571](https://github.com/Motoki0705/tennis-lab/pull/2571)

- イベント照合は1対1の一致数を最大化し、同数なら絶対時刻誤差の合計を最小化します。レビュー例は2件とも一致し、F1は1.0です。BLCSの既存評価も同じ照合処理を使います。
- FPS照合の境界では1 ULPの丸め誤差を許容します。60 fpsの重みに対して、許容幅0.06 Hzを明示した59.94と60000/1001を受け付け、許容幅未指定と許容幅外のケースを拒否することを確認しました。

合成test 128ラリーを再集計し、全クラスの適合率・再現率・F1・時刻誤差が学習時の記録と一致しました。F1はショット0.932、バウンド0.986、ネット接触0.857です。`clip_000` のJSONとNPZも再実行でバイト一致しました。 [再集計と記録PR #2574](https://github.com/Motoki0705/tennis-lab/pull/2574)

採用checkpointは `best.ckpt` で、SHA-256は `b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0` です。修正後の実装PRはCI待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2571,"head":"4afdccc7c348a946cc5bb86aac583ae220ce652f","resume_on_success":false}}