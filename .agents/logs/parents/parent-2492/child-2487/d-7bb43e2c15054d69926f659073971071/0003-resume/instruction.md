# 再開（親 #2492 から）: ONNX変換jobの結果

前ターンの最終行 `PARENT_RESULT` は閉じ括弧が1つ足りないJSONで、親CLIが待機を登録できなかった（`PARENT_RESULT: invalid JSON`）。宣言されたlocal jobは親が確認した。最終行は1行の正しいJSONにすること。

- `20261124T114215Z_5ea907a8d63a_rtdetrv4-l-onnx`: state=done、exit_code 0、38.5 s
- `20261124T114215Z_9940a8133486_rtdetrv4-x-onnx`: state=done、exit_code 0、50.3 s

ログは `[local path]` にある（TorchScript ONNX exporterの警告あり）。生成物を検証し、指示どおり #2495 の採点器で L/X × PT32/PT16/TRT16 の精度を、#2496 のharnessで速度・VRAMを計測する（GPUはtraining queue経由）。

#2496 の固定速度入力がまだ無い場合は、精度の計測を先に進め、速度は「#2496 の固定入力待ち」と明記する。速度だけ独自の入力で測って #2496 の値と並べることはしない。

待機で終える前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してから終えること（両PRの最新headが含まれていないと、自動再開は拒否される）。
