# 担当: #2335（親 #2278 Phase 1 追加: CLIP-ReID・GVHMRのバッチサイズ掃引）

Issue #2335 の本文（Acceptance checklist を含む）が正本。親 #2278 の合意・対象外・共通原則も守る。

## 進め方（作業単位と時間上限）
1. バッチ掃引harness（上限3時間）: #2297 の harness をバッチ次元に一般化する（後方互換、既存テストを通す）。TRTエンジンはバッチ次元を動的にし、原則 min/opt/max = 1/16/最大バッチの単一profile。後続の #2336（ViTPose-H・HMR2・DINO）が同じharnessを再利用する前提で作る。
2. CLIP-ReID batch 1/4/8/16/32 × 条件0/1/2（上限4時間）。
3. GVHMR B=1/2/4 × 条件0/1/2（上限5時間）。#2305 の `knowledge/runs/run-i2305-gvhmr-tensorrt/` を土台にする。系列の束ね方（mask／同一長のみ）を決めて記録する。
全体の上限は12時間。ncuは代表バッチ（原則16、GVHMRは4）だけ。

## 作業場所・基準
- worktree は parent CLI が作る（基準 origin/main b392f45ee、#2305 マージ後）。メインworktreeは変更しない。
- `.venv`・`data`・`ckpt` はメインへのsymlink、`outputs` は永続出力。共有 `.venv` には何も install しない。TensorRT等は専用環境 `$HOME/.venvs/tennis-lab-trt` を使う。
- 計測手順・指標・統計量は `knowledge/runs/run-i2297-clipreid-tensorrt/PROTOCOL.md` に従う。harness はそのディレクトリで一般化し、コードを複製しない。新しいrunのデータ・結果・knowledgeノードは新しい `knowledge/runs/<run-id>[local path]` に置く。`src/` は変更しない。knowledge の記録は knowledge-control skill に従う。
- 並行して #2303（ViTPose-H・HMR2）と #2304（DINO）のワーカーが同じ共有harness（convert.py・profiles.py・runtime.py）を変更中。変更は最小限・後方互換にし、PRは小さく保つ。
- GPU・大メモリ処理（export、engine build、計測、ncu/nsys）はすべて training queue 経由（`TRAINING_QUEUE_DIR=[local path]`、worktreeのcwdから add、`--resource all`）。queue worker の start・serve はしない。WSLメモリ上限40GB・VRAM 16GBを超えるとホストが固まるので、最大バッチは余裕を持って決め、超える場合は理由を記録する。
- **待機の受け渡し**: 投入したら、出力された `.job` 付きのIDを最終メッセージ末尾の `PARENT_RESULT` の queue 待ちで宣言してターンを終える。PR作成・push後は CI 待ちを宣言する（成功後にドラフト解除が残るなら `resume_on_success: true` と `next_step`）。自分で待たない・照会ループを回さない。投入準備ができたら同じターンで投入する。
- 静かなフォールバック禁止: 変換失敗・範囲外バッチ・VRAM不足・指定精度で実行されない場合は非ゼロで止め、根拠付きで記録する。小さいバッチへ自動で落とさない。

## 止めてよい条件
(a) Acceptanceをすべて満たした、(b) 人間の操作が要り、残る全作業がそれに依存する、(c) 共有 `.venv`・依存定義・`src/` を変えないと進めない、(d) 親 #2278 の合意範囲の判断が要る。

## 返す成果
- `Closes #2335` を本文に書いた日本語PR。対象HEADの `Python` CI成功後にドラフトを解除する。
- 最終報告: Acceptanceごとの達成/未達と証拠パス、モデル×条件×バッチの要約表（中央値[local path]、throughput、Peak VRAM、kernel数、単体精度）、代表バッチのncu指標とbatch 1との比較、未確認事項、E2E評価と #2336 への引き継ぎ。
