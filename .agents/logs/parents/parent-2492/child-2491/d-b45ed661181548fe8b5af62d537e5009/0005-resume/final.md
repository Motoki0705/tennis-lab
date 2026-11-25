FastHMR CameraHMRの実重みをRTX 5060 Tiで検証しました。`clip_003` の同じ8フレームで、PT32・PT16ともに有限なSMPLメッシュを生成しました。カメラ座標のCOCO17関節差は **P95 45.12 mm** で、この試行では共通基準の10 mmを超えました。生予測2点と照合手順を保存しました。[実測記録PR #2534](https://github.com/Motoki0705/tennis-lab/pull/2534)

代表6 clipの固定入力1,711フレームを保存済みGVHMR成果と照合しました。`codex/parent-2492-issue-2491-primary` の `9e7448a34a81dc36e62512f8c3185d48a67beb9b` をpushし、標準recipeのキャッシュ互換を含む37件が通過しました（20.04秒）。[実装PR #2501](https://github.com/Motoki0705/tennis-lab/pull/2501)

代表6 clipのPT32・PT16推論と配置、実入力を使うTRT16変換試行をキューに登録しました。同期比較動画は予測の回収後に作成します。速度・VRAMの計測と代表6 clipのparity判定は、#2497の修正版待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204434740687790252_2026-11-25T00-07-39-312209747Z_2393387_i2491-fasthmr-trt16-export.job","0-8204434740417409116_2026-11-25T00-07-39-582590883Z_2393349_i2491-fasthmr-quality-6clips.job"]}}