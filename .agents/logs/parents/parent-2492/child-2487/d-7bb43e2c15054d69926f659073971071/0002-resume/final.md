L/X とも公式配布の重みをそのまま読み込み、`clip_003 / cam0 / frame 120` で CPU 推論が通りました。旧形式に対応するため `mmap` を外し、両ファイルの SHA-256 を固定しました。スコア 0.4 超の人物検出は L が10件、X が9件でした。

修正は `codex/parent-2492-issue-2487-rtdetrv4` の `9fe7b3987e52b7dd0f0d255e633bb310b0120861` にコミット済みです。#2498 と #2497 の最新 HEAD を含む状態で、29件のテストと pre-commit が通りました。

L/X の ONNX 変換を再登録しました。GPUでの精度・速度・VRAMは未計測です。速度計測には #2496 の固定入力が必要です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T114215Z_5ea907a8d63a_rtdetrv4-l-onnx","20261124T114215Z_9940a8133486_rtdetrv4-x-onnx"]}