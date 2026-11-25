RTMPose-M/X の PT16・TRT16 速度計測を、モデル別の2ジョブとして共有GPU queueに登録しました。#2497 の固定入力と同期付きharnessで batch 1・4・8 を測る計12条件です。parity は `PT32未計測` と記録します。

`codex/parent-2492-issue-2490-rtmpose` に `645bf735dc4a183c46ce1a9448bbd8eee8d0ec30` をpushしました。#2497 のマージコミット `59faa32a28d2f54e6492abdba2429846fa0397e3` を含みます。

影響範囲の116テストが14.30秒で通過し、pre-commitも通過しました。固定入力の計時順と、M/Xの重み・TRT16 engineのSHA-256を照合済みです。速度・VRAMはqueue完了後に条件別の出力を確認します。 [起動済みのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45590029165)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204399428034176235_2026-11-25T09-56-11-965823764Z_2585778_i2490-rtmpose-m-low-precision-speed.job","0-8204399427688229043_2026-11-25T09-56-12-311770956Z_2585722_i2490-rtmpose-x-low-precision-speed.job"]}}