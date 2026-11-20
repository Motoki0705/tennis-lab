# 続き（同じsession・branch・worktree）: まとめてPRを出す

親の判断:
1. **INT8のncuは追わない**: `i2304-dino-grid-3a-ncu-v7` は exit_code 3 で失敗した（ログ: `.training_queue/logs/0-8204816680926065162_2026-11-20T14-01-59-073934837Z_1137002_i2304-dino-grid-3a-ncu-v7.log`）。ログでkernel replay・application replayのどこで止まったかを確かめる。条件3aのDRAM・stall・Tensor Coreは「未採取」として、45分の上限で複数回打ち切られたことなどを根拠付きで記録する。速度・VRAM・nsys・精度は取得済みなので、条件3aは計測済みとして扱ってよい。これ以上のGPUジョブは投入しない。
2. **PTQ設定**: #2303（PR #2334）がmainに入り、PROTOCOL.md の条件3a/3bの定義がViT系の新しい設定（3a: attention出力投影だけをINT8 max、3b: 重み付き線形層4種をFP8 max）に変わった。DINOの3a/3bは、計測時点の #2297 の設定で取った結果として、設定の違いを明記して記録する（#2305 GVHMRと同じ扱い）。新しい設定での再計測は、このPRではしない。
3. **MSDAの方式**: 条件2はGridSample版（57.0 ms）を正式な値とし、TensorRTプラグイン版（41.5 ms）を比較として並べる。方式の採否は親 #2278 で人間が決めるので、どちらかを結論として採用しない。両方式の速度・精度・実装と運用の負担（プラグインのビルドと配布、PTQとの両立）を判断材料として書く。
4. **mainの取り込み**: mainには #2305（PR #2327）と #2303（PR #2334）が入り、共有harness（`knowledge/runs/run-i2297-clipreid-tensorrt/` の convert.py・profiles.py・runtime.py・PROTOCOL.md）が変わっている。origin/main を取り込んで競合を解消し、#2297・#2303・#2305 の既存テストとDINOのテストを通す。共有harnessの変更は最小限・後方互換にする。

そのうえで、knowledgeのrunノードを作り（knowledge-control skill）、pre-commitを通してcommitし、`Closes #2304` の日本語ドラフトPRを作ってpushする。最後に `PARENT_RESULT` でCI待ちを宣言して終了する（`resume_on_success: true`、next_step: ドラフト解除と最終報告）。最終報告には次を書く。
- Acceptanceごとの証拠
- 5条件の要約表とプラグイン版の比較
- 3aのncuが未採取である根拠
- 未確認の事項
