正式レビューのmajor指摘2件を `codex/person-stage-speed` の `a9f9ff820` に反映しました。PRはReady for reviewです。 [PR #2240](https://github.com/Motoki0705/tennis-lab/pull/2240)

- 基準mainが保存した31段の成果物を、既定FP32と明示FP32の両方で再利用できることを確認しました。通常実行は全段 `cached`、load専用は全段 `loaded` でした。dataset公開identityも一致し、BF16・FP16は別キーです。ソースの互換扱いは照合済みの差分に限定しました。
- clip_000のカメラ間人物ID差2件を、追跡profileと対応用CLIPの実行deviceを同時に変えた観測として訂正しました。姿勢なしprofileの影響は、固定特徴で比較した追跡・選手選別までを根拠としています。 [knowledge 000050](https://github.com/Motoki0705/tennis-lab/blob/a9f9ff820/knowledge/nodes/tennis_scene/000050-run-person-stage-precision-meiji-20261118.md)

関連CPUテスト1,559件、pre-commit、対象HEADのCI全14ジョブとKnowledge libraryが通過しました。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44283666877)・[Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44283666930)

標準の既定はFP32です。meiji_4cam本番ではDINO・ViTPoseともにFP16とする決定を記録しました。両モデルを同時にFP16にした条件のGPU実測は未実施です。