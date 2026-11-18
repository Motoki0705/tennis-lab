a5は両版とも1010frameを完走し、CUDA異常はありませんでした。**Ball差分が事前条件を超えたため比較は不合格、[PR #2182](https://github.com/Motoki0705/tennis-lab/pull/2182)はDraftを維持しています。**

- Court/Ball/DINO/ViTPoseの読込は各3→1、CLIPは4→1。8resourceのload/unloadは各1回、同時保持最大2、未解放0。
- NVML観測最大は4533→4568 MiB。Torch allocated peakは両版とも約3.160 GB。
- 全mask・IDと人物系出力は完全一致。Ball検出は最大0.129 px、最終3D位置はL2最大0.362 mmの差があり、`atol=1e-6 / rtol=1e-5`を超えました。原因は未確定です。
- [knowledgeに全結果を保存](https://github.com/Motoki0705/tennis-lab/blob/23ea27f29/knowledge/nodes/tennis_scene/000052-run-i1751-runtime-gpu-a5.md)。branchは`codex/scene-inference-runtime`、HEADは`23ea27f29`です。

推論source `1bef628a9` のCIは全成功。最新HEADのKnowledge CIは成功、[CPU CIは実行中](https://github.com/Motoki0705/tennis-lab/actions/runs/44151638309)です。

**このBF16差分をC3受入として許容するか、親の判断が必要です。** 追加GPU実行は行っていません。