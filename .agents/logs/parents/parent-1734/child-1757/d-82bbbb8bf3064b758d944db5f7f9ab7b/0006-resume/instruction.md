# #1757: 条件2とCLIPの使い回しを記録し、#2272を取り込んで、条件3を投入する（親 #1734）

## 状況
- 条件2（`reuse-heartbeat-1`）と、CLIPの使い回しの条件（`embedding-reuse-heartbeat-1`）が、両方とも `status: complete` で完走した。HEADは `795d63df`。
- 親が読んだ全体の時間:

  | 条件 | first | warm | warm |
  |---|---|---|---|
  | reuse | 1862.8 s | 1797.1 s | 1808.4 s |
  | embedding_reuse | 1908.0 s | 1823.4 s | 1815.3 s |
  | 逐次（条件1、`8c91bf074`） | 2118.0 s | 2031.5 s | 2005.1 s |

- PR #2272（#1755、batch推論）が、レビューのmajor 4件を直したうえでmainにmergeされた。修正は次の4つ。
  - ViTPoseのUDPで、無効な関節を除外する
  - Courtで、行単位に異常を分離する
  - X3Dの予約にFP32の入力を加え、上限を3 GiBにする
  - ClipStoreで、キャッシュのヒット時にもchecksumを検証する
- queue workerは、親が独立して起動している（PID 1419109）。queueに走っているjobはない。

## やること
1. 2つの条件の結果を、run noteとknowledgeのnodeに記録する。
   - 段階ごとの内訳と、資源のピーク（GPUのallocated/reserved、RSS）を書く。
   - 出力を、事前に決めた許容差（`tolerances.json`）で、逐次の出力と照合する。
   - court_sideの判定と、CLIPの使い回しの命中数も書く。
   - **CLIPの使い回しのほうが、reuseよりわずかに遅い。** これが計測のばらつきなのか、本当にかかっている費用なのかを、段階ごとの内訳から判断して書く。
   - 結果が一致しなければ、そう書く。
2. `origin/main`（#2272を含む）を、このbranchに取り込む。衝突は、mainの実装を正として解く。CPUのテストとpre-commitを通す。
3. 取り込んだHEADで、**条件3（#1755のbatch推論）** を、新しいrun名でqueueに入れる。
   - `start` も `serve` も実行しない。`status` で `worker: RUNNING` を確かめてから `add` すること。
   - 監視の方法は、これまでと同じにする。heartbeatと外部の上限で見て、`nvidia-smi` は呼ばない。
   - batchの設定（batch sizeとメモリの上限）は、PRの既定値を使う。その値を報告する。
4. 本計測に入り、heartbeatが更新されていることを確かめたら、このターンを終える。止まった場合や、CUDAの異常が出た場合は、再試行せずに報告する。
5. PR #2273の状況欄を更新する。job ID・ログの場所を報告する。背景にプロセスを残さない。
