# 続き（同じsession・branch・worktree）

前のターンで宣言した queue 待ちの3件は、次の状態になった。親は2026-11-22にWSLが落ちて再起動したことを確認しており、待機の監視が消えたため手動で再開する。結果は各ジョブのログ・出力で必ず確かめること。

- `i2336-vitpose-timing`: failed、`exit_code=124`。c0-b1〜b32 と c1-b1〜b16 は exit 0 で検証済み。その後に時間切れになった（どの段階で切れたかはログで確かめる）。
- `i2336-hmr2-timing`: WSLの再起動で中断した。queue 上は `state=running` のまま残っているが、wrapper のプロセスは存在しない。c0・c1 の全バッチは exit 0 で検証済み。`prepare-2` は exit 0、`build-2` は exit 1、`prepare-3a` は exit 1 で、そこで中断した。
- `i2336-dino-timing`: 再起動後に実行され、終了した（exit_code=1）。c0・c1 の b1/2/4、`export-2`・`prepare-2` は exit 0。`build-2`（GridSample版）と `build-2-plugin` が、どちらも exit 1。

## 親からの指示
- 3モデルとも TRT のエンジン作成（build-2）で失敗している。まず、共通の原因（専用環境 `$HOME/.venvs/tennis-lab-trt`・TensorRTの状態、ディスク、build の時間上限、profile の設定など）か、モデル固有の原因かを、ログを根拠に切り分ける。
  - WSLのクラッシュの前後で、ストレージ（`.worktree-outputs` が E: 側への symlink に変わった）や環境が変わっている可能性がある。確かめた事実と推測は分けて書く。
- hmr2 の中断したjobを、自分で終端状態に書き換えない。queue の state は親が扱う。再実行が必要なら、新しい名前で投入する。
- ViTPose-H の時間切れは、ジョブの分け方（条件・バッチごとに分ける）か、時間上限を根拠付きで見直す。
- 修正したら、同じターンで残りの計測を投入し、出力された job ID を `PARENT_RESULT` の queue 待ちで宣言して終わる。自分で待たない。
- 当初の時間上限・止めてよい条件・Acceptanceは引き続き有効。
