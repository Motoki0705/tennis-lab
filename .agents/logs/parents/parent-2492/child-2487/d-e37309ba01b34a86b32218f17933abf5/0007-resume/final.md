RF-DETR-L の初回4条件をすべて検証しました。固定96画像で各300反復を完了し、batch 1 の stage wall P50 は PT16 が **58.05 ms/画像**、TRT16 が **33.97 ms/画像**でした。VRAM・生レポート・ログ・実行コミット・資産のSHA-256を記録しました。[初回の実行証拠](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7162696014)

初回はPyTorchのCPUスレッド数がRF-DETRで2、DINOで4でした。DINOと同じPyTorch 4・OpenCV 1に揃え、PT16・TRT16のbatch 1・2を再投入しました。両batchのTensorRT planを再利用し、parityは「PT32未計測」です。

`codex/parent-2492-issue-2487-rfdetr` に `8875cd0c17459660ff4f9f491c3a5fb1b965bfb5` をpushしました。対象テスト62件、pre-commit、knowledge検証が通過しました。DINO比較表は再計測結果の受領後に更新します。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45615971887)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204389561547152306_2026-11-25T12-40-38-452847693Z_3832713_i2487-rfdetr-l-pt16-b1-matched-speed.job","0-8204389561265977834_2026-11-25T12-40-38-734022165Z_3832782_i2487-rfdetr-l-pt16-b2-matched-speed.job","0-8204389560981567683_2026-11-25T12-40-39-018432316Z_3832852_i2487-rfdetr-l-trt16-b1-matched-speed.job","0-8204389560691772196_2026-11-25T12-40-39-308227803Z_3832920_i2487-rfdetr-l-trt16-b2-matched-speed.job"]}}