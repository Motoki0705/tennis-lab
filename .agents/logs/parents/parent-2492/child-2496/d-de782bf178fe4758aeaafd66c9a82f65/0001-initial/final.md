#2496 の共通推論ベンチマークをドラフトPRにしました。PT32・PT16・TRT16で、CUDA eventによるforward時間、同期済みのstage wall time、P50/P95、items/s、VRAM、parityを記録します。既定はbatch 1を含む掃引、warm-up 30回、100回×3ブロック、TF32無効です。 [PR #2497](https://github.com/Motoki0705/tennis-lab/pull/2497) · [使い方](src/utils/inference_benchmark.md)

branch は `codex/parent-2492-issue-2496-primary`、完全HEADは `8dbafd61724b67265feea99e92cd6b4852efd1f4` です。CPUテスト49件（新規13件、既存36件）とpre-commitが通過しました。OOM、変換失敗、parity不成立、別backendへの切替拒否を確認済みです。

固定6 clipの入力manifestとSHA-256、実GPUでの速度・VRAM、既存計測値との照合、knowledgeへの記録は残っています。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2497,"head":"8dbafd61724b67265feea99e92cd6b4852efd1f4","resume_on_success":true,"next_step":"代表6 clipの固定frame・人物cropをIDとSHA-256付きで保存し、body_view_selectionと既存TensorRT資産を照合して、現行モデルのPT32/PT16/TRT16計測を共有training queueへ投入する。"}}