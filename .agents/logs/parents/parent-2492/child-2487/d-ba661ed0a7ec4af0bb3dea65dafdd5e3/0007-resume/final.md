[#2499](https://github.com/Motoki0705/tennis-lab/pull/2499) に `65d6349a5bb1b6c4f88f4fbbc894cc8e6f5b480a` をpushし、一体計測を共有queueの1ジョブに登録しました。

修正版 #2497 harnessで、固定96画像をPT16・TRT16 × 画像batch 1・2の4条件で測ります。DINOのscore 0.30以上の全検出を、1.2倍の正方形cropにしてViTPose-Hへ渡します。flipあり・最大8 cropは初回と同じです。parityは「PT32未計測」とし、初回PT32の590.33 ms/画像は旧harnessの暫定値として保存しています。

対象pytest 44件、pre-commit、knowledge検証が通過しました。固定画像とTensorRT planのSHA-256を照合済みです。新しい速度・VRAMの結果はqueue待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204398523191235973_2026-11-25T10-11-16-808764026Z_2693116_i2487-dino-vitpose-joint-synchronized.job"]}}