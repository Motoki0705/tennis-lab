#2697の本学習と終端評価を共有GPU queueに投入しました。3条件とも **512×288、seed 42、4,700更新、128フレーム／更新**です。終端チェックポイントを固定validation 117,358件と #2637 の144アンカーで採点します。

720pのプローブではT=1がVRAMの安全上限に達しました。512×288は全3条件が通り、評価込みの実測ピーク予約量は4.35 GiBでした。プローブから全評価までの合計は、20%の余裕を含めて約5時間56分と見積もっています。[#2697の実験設計コメント](https://github.com/Motoki0705/tennis-lab/issues/2697#issuecomment-7222572419)

実行版は `bd0d76a14cdf22cc5b7f928d19831efc89c49b5a` です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204205277883712570_2026-11-27T15-52-02-116287429Z_1813138_i2697-ball-temporal-training-s42.job"]}}