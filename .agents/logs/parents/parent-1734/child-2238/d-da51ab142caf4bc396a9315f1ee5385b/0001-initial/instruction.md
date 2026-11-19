# #2238 の作業指示（親 #1734）

issue #2238 の本文が仕様です。まず読んでください。補足は次のとおりです。

- **背景の数値**（#1752の実測。knowledge `run-i1752-association-stop-20261118` の `prefix-summary.json`）: clip_000の4cameraで、ViTPose 514秒（1424回。1回のbatchは10〜24人分で、flip_testあり）、DINO 378秒（1424枚、batch 1、fp32）、CLIP-ReID 28秒。人物対応までの合計は1201秒。
- **優先順位**: 効果が大きく、品質への影響が小さいものから測る。
  1. ViTPoseのbf16/fp16のautocast
  2. DINOのbf16/fp16（独自CUDA opの扱いを明示する）
  3. ViTPoseのflip_testを外す
  4. 姿勢推定の対象を絞る案の調査
  まずは1と2の速度と差を測り、途中の結果をknowledgeに残す。
- **batch化は#1755の範囲**。DINOのbatch APIは#1750（merge済み）で使えるので、速度の比較にbatchの大きさを含めてもよい。ただし、既定の経路は変えない。
- **GPU**: 必ず共有training queue経由で使う。CUDAのハングかエラーが起きたら、再試行しない。全threadのstackとGPUの状態（`nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE`）を保存して停止し、報告する。11/18の未明に、DINOの `DeformableTransformer.forward` で180秒の停止が1回起きている（knowledge 000051、branch `codex/scene-inference-runtime`）。
- **同時に動くGPUジョブ**: #1752の3clipの全段の実行が、queueで先に走っている。queueの順番を守る。
- 完了したら、PRを作ってCIを確認し、Ready for reviewにして終了する。採用案（既定値を変えるか）は人間が決めるので、既定はfp32のままにしておく。merge・closeは親が行う。
