[実装PR #2571](https://github.com/Motoki0705/tennis-lab/pull/2571) を作成しました。HEADは `e4438667dd41162c6f4972a92146eb9b59bbba54` です。最終HEADで統合テスト33件とpre-commitが通過しました。

新データ版 `data/ball_refiner/single_object_net_v1` は全1,280ラリーです。ネット接触は640件で、白帯を越えてプレーが続く球と打者側に落ちて終わる球が320件ずつでした。生成時に全軌道の再積分をビット照合し、保存後は全ファイルのSHA-256と全接触の内訳を確認しました。

固定したデータ版で、回帰モデルの4,000ステップ学習を共有queueの `half` 枠に登録しました。学習完了後に合成testのクラス別指標、`clip_000` の推論結果、checkpointのSHA-256を確認し、ナレッジ記録PRに残します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204370269760775669_2026-11-25T18-02-10-239224330Z_1956258_ball-refiner-shot-bounce-net-s42.job"]}}