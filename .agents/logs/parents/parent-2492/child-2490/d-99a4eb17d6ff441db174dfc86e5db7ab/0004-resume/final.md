[draft PR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500) にViTPose-H・RTMPose・PMPoseの18条件を統合しました。保存済み予測を更新版の共通採点器で再採点し、OKS・PCK・pixel誤差・Recallを含む全集計値が一致しました。

PT32の平均OKSはViTPose-Hのflipありが **0.77777**、なしが **0.77594**、PMPose-Hが **0.78084** でした。ViTPose-HのflipありTRT16はPT32との座標差がP95で **0.024 px**、最大で **90.8 px** でした。 [精度表](https://github.com/Motoki0705/tennis-lab/blob/ff77e3db0411432f2952a5ce752eeb6482441eed/knowledge/runs/run-meiji-topdown-pose-comparison/accuracy.csv) · [185フレームの同期比較](https://github.com/Motoki0705/tennis-lab/blob/ff77e3db0411432f2952a5ce752eeb6482441eed/outputs/pose-comparison/clip_003-cam2-pt32.mp4)

ローカルテスト61件、pre-commit、knowledge検査が通過し、新HEADのCIが実行中です。Sapiens2-5BのFP32はCUDAロードまで成功し、forwardは確認待ちです。難条件別集計、全代表クリップの可視化、速度・VRAM・Paretoの統合が残っています。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"ff77e3db0411432f2952a5ce752eeb6482441eed","resume_on_success":true,"next_step":"難条件別の関節集計と同期比較を拡張し、Sapiens2担当の精度・変換結果を統合する。速度再計測は #2497 修正版と親の再投入指示を確認して進める。"}}