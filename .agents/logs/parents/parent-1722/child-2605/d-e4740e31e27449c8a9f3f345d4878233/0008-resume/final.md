SceneResult v3と `event_detection` の3方式をPR #2610に実装しました。`codex/parent-1722-issue-2605-primary` の `555a246aabb7201959e3bb28cc6f40079e62b5c3` でPython CIが成功し、同じHEADの実行証拠をPR本文に記録しました。 [PR #2610](https://github.com/Motoki0705/tennis-lab/pull/2610)・[CI結果](https://github.com/Motoki0705/tennis-lab/actions/runs/45768027741)

- `multiview-v3` の69クリップをCPUで再生成し、検証済みの成果物を配布先に反映しました。全69クリップのv2配列がビット一致し、16,302ファイルのhard linkと元注釈69件のSHA-256を確認しました。旧版は `.multiview-v3-59a62db3c` に保管しました。
- 全913イベントは打球421、バウンド457、ネット35。判定不能は1,885フレームです。train／val／testは47／11／11で、指定の採点用6クリップがtestに入っています。 [配布manifest](https://github.com/Motoki0705/tennis-lab/blob/555a246aabb7201959e3bb28cc6f40079e62b5c3/data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3/manifest.json)
- 実クリップ `clip_057` でannotation・rule・refinerを再実行し、全方式でv2配列のビット一致を確認しました。

統合レビューとマージを親に引き渡します。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}