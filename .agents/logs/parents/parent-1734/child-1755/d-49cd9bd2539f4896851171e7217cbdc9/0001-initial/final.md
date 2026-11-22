[PR #2272](https://github.com/Motoki0705/tennis-lab/pull/2272) を Ready for review にしました。ブランチは `codex/video-batch-batching`、HEADは `516bd33de998c42155a7c6fedef3a3df6dddcbeb` です。PRは `main` を対象とし、依存する一括CLIの #2227 の差分を含みます。

Court・DINO・ViTPose・CLIP・HMR2・Ballの横断バッチを実装しました。`batching.enabled=true` で有効になり、B=1、同時source数、queue、作業メモリ、全ClipStoreの共有cacheに上限を設定できます。充填率とpassごとのモデル読込・動画open回数を記録します。Ballは時間窓のB軸をまとめ、GVHMRはsegment単位で実行します。[設定と初期値](https://github.com/Motoki0705/tennis-lab/blob/516bd33de998c42155a7c6fedef3a3df6dddcbeb/src/tennis_scene/pipeline/README.md#L64)

最終HEADで関連CPUテスト740件、pre-commit、Python CI、Knowledge CIが通過しました。CPUテストではB=1との書き戻しの一致、メモリ上限、OOM時の停止、選手のみの2clipの公開と再利用を確認しました。[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44470954135)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44470954103)

#1757のGPU事前計測は `nvidia-smi` の10秒タイムアウトで停止しました。固定3clipでの出力一致、速度、VRAM・RAM・一時ディスクの実測と、採用batch値の確定が残っています。[計測PR #2273](https://github.com/Motoki0705/tennis-lab/pull/2273)