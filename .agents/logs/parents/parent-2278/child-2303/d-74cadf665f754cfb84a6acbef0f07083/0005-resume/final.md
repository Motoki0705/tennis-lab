HMR2の5条件すべてで実測が揃いました。TensorRT FP16／INT8／FP8の中央値は **6.54／6.57／5.40 ms** で、いずれも単体精度の暫定基準を満たしました。INT8の追測は同じエンジンで341カーネルを取得し、Nsight Systemsの件数と一致しました。[条件別の実測](https://github.com/Motoki0705/tennis-lab/blob/a848ab29fe7f6fcaad4a764a564a722d6fa51c54/knowledge/runs/run-i2303-vit-tensorrt/metrics.json)

ViTPose-HのINT8は中央値 **8.69 ms**、座標差P95 **0.079 px** でした。有効4,012関節のうち1件でヒートマップのピークが移り、座標差 **23.35 px** が暫定上限5 pxを超えました。GPUカウンタは300カーネル中228件で30分の時間制限に達しました。[検証記録](https://github.com/Motoki0705/tennis-lab/blob/a848ab29fe7f6fcaad4a764a564a722d6fa51c54/knowledge/runs/run-i2303-vit-tensorrt/validation.json)

結果は `codex/trt-vit-pose-hmr2` の `a848ab29fe7f6fcaad4a764a564a722d6fa51c54` に記録し、knowledge検証を通しました。残る `i2303-vitpose-3a-ncu-r5` は同じエンジンと代表入力でカウンタだけを60分枠で取得します。再開後はその結果を照合し、比較表とrunノードを完成させてPRを作成します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204828400464429831_2026-11-20T10-46-39-535570168Z_4184786_i2303-vitpose-3a-ncu-r5.job"]}}