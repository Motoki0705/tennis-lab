# 担当: #2336（親 #2278 Phase 1 追加: ViTPose-H・HMR2・DINOのバッチサイズ掃引）

Issue #2336 の本文（Acceptance checklist を含む）が正本。親 #2278 の合意・対象外・共通原則も守る。

## 進め方（作業単位と時間上限）
1. **準備（上限2時間）**: #2335（PR #2349）のバッチ掃引harnessの使い方を読む（`knowledge/runs/run-i2335-trt-batch-sweep/README.md#2336での再利用`）。そのうえで、ViTPose-H・HMR2（#2303: `run-i2303-vit-tensorrt`）とDINO（#2304: `run-i2304-dino-tensorrt`）の固定入力・変換に接続する。harnessは複製しない。
2. **ViTPose-H・HMR2（上限6時間）**: batch 1/4/8/16/32。
   - ViTPose-Hはflip test込みで、エンジン入力は2B。throughputは処理したcrop数（B）で数える。
   - 条件は0・1・2と、batch 1で単体精度が許容内だったもの。HMR2は3a・3b（#2334の設定）、ViTPose-HはINT8・FP8とも不合格なので対象外。
3. **DINO（上限5時間）**: shape bucket 750×1333 で batch 1/2/4。VRAMに収まらないバッチは、エラー出力を根拠に記録する。
   - 条件は0・1・2。INT8・FP8は単体精度が草案の範囲外なので対象外。
   - 条件2はGridSample版を正式な値とし、プラグイン版も条件2で測って並べる。
4. 全体の上限は14時間。ncuは代表バッチ（ViTPose-H・HMR2は16、DINOは4か、収まる最大のバッチ）だけで取る。

## 作業場所・基準
- worktree は parent CLI が作る。基準は origin/main 6bd10cac9（#2349 マージ後）。メインworktreeは変更しない。
- `.venv`・`data`・`ckpt` はメインへのsymlink、`outputs` は永続出力。共有 `.venv` には何も install しない。TensorRT等は専用環境 `$HOME/.venvs/tennis-lab-trt` を使う。
- 計測手順・指標・統計量は `knowledge/runs/run-i2297-clipreid-tensorrt/PROTOCOL.md` に従う。新しいrunのデータ・結果・knowledgeノードは新しい `knowledge/runs/<run-id>[local path]` に置く。`src/` は変更しない。knowledge の記録は knowledge-control skill に従う。
- GPU・大メモリ処理はすべて training queue 経由（`TRAINING_QUEUE_DIR=[local path]`、worktreeのcwdから add、`--resource all`）。queue worker の start・serve はしない。WSLメモリの上限40GB・VRAM 16GBを超えるとホストが固まるので、最大バッチは余裕を持って決める。
- **待機の受け渡し**: 投入したら、出力された `.job` 付きのIDを、最終メッセージ末尾の `PARENT_RESULT` の queue 待ちで宣言してターンを終える。PR作成・push後はCI待ちを宣言する（成功後にドラフト解除が残るなら `resume_on_success: true` と `next_step`）。自分で待たない。投入の準備ができたら同じターンで投入する。
- **静かなフォールバック禁止**: 変換失敗・範囲外のバッチ・VRAM不足・指定精度で実行されない場合は非ゼロで止め、根拠付きで記録する。小さいバッチへ自動で落とさない。同じビルド失敗が方式を変えて2回続いたら追わず、「不可」として根拠を記録する。

## 止めてよい条件
(a) Acceptanceをすべて満たした、(b) 人間の操作が要り、残る全作業がそれに依存する、(c) 共有 `.venv`・依存定義・`src/` を変えないと進めない、(d) 親 #2278 の合意範囲の判断が要る。

## 返す成果
- `Closes #2336` を本文に書いた日本語PR。対象HEADの `Python` CI成功後にドラフトを解除する。
- **最終報告**:
  - Acceptanceごとの達成/未達と証拠パス。
  - モデル×条件×バッチの要約表（中央値[local path]、throughput、Peak VRAM、kernel数、単体精度）。
  - 代表バッチのncu指標と、batch 1との比較。
  - 未確認事項と、E2E評価への引き継ぎ。
