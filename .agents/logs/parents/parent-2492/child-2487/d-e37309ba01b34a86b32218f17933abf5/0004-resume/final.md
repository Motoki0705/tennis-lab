RF-DETR-LのPT32・PT16・TRT16を、#2496の固定96画像とDINO基準と同じbatch 1・2、warmup 30回、300反復で測る6ジョブを共有queueに登録しました。速度・VRAM・parityの結果を受け取り次第、DINO基準と統合用の比較表にまとめます。

更新版の採点器で保存済みの6,844画像を3条件とも再採点し、Recall@0.50を含む全採点値が一致しました。[再採点記録](https://github.com/Motoki0705/tennis-lab/blob/73f2f9969c97abd5f6c346eb2cc6fd4dbe7c7a46/knowledge/runs/run-meiji-rfdetr-large-accuracy/canonical-rescore.json)

計測HEADは `73f2f9969c97abd5f6c346eb2cc6fd4dbe7c7a46`。#2497の最新head `14ca88f8a` を取り込み済みで、対象テスト46件とpre-commitが通過しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204459650988321183_2026-11-24T17-12-29-011678816Z_3667397_i2487-rfdetr-l-pt32-b1-speed.job","0-8204459650715102407_2026-11-24T17-12-29-284897592Z_3667465_i2487-rfdetr-l-pt32-b2-speed.job","0-8204459650443192095_2026-11-24T17-12-29-556807904Z_3667534_i2487-rfdetr-l-pt16-b1-speed.job","0-8204459650127155170_2026-11-24T17-12-29-872844829Z_3667601_i2487-rfdetr-l-pt16-b2-speed.job","0-8204459649807334227_2026-11-24T17-12-30-192665772Z_3667674_i2487-rfdetr-l-trt16-b1-speed.job","0-8204459649535148361_2026-11-24T17-12-30-464851638Z_3667747_i2487-rfdetr-l-trt16-b2-speed.job"]}}