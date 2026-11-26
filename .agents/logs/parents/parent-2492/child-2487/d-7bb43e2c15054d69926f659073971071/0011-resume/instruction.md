# #2487 rtdetrv4: TRT16 は打ち切り、PT16 の速度をスレッドを揃えて測り直す（親の判断）

前回の報告（`11631d3be`、`trt16-fp32attn` は全8ケースで logits と BBox が NaN）を受け取った。記録を承認する。

## 判断
- RT-DETRv4 の TensorRT は、これ以上調べない。元の TRT16（Recall ほぼ0%）と `trt16-fp32attn`（NaN）の両方を「変換の不具合、原因未特定」として記録したままにする。比較図では、精度を Pareto 計算から外す扱いを続ける。最初に NaN が出る演算の特定は、採用候補になったときの課題として残す。
- PT16 の速度は比較図に載る。ところが、統合担当の報告では、RT-DETRv4 の計時は CPU 2スレッドで、DINO と RF-DETR-L は4スレッドだった。EdgeCrafter はすでに揃えて測り直した（`60af57ee6`、PyTorch 4・OpenCV 1スレッド）。

## やること
1. DINO の速度基準（#2496 の `run-person-model-speed-baseline` と、#2487 統合担当の DINO の計測）と同じホストスレッド設定を、記録から確かめる。それに合わせて、RT-DETRv4 L/X PT16 の batch 1・2 の速度を測り直す。harness は main の同期込み版、固定入力と反復数は前回と同じ、parity は「PT32未計測」。古い2スレッドの値は消さずに残し、スレッド設定が違うことを書く。
2. queue のログの保持。`.training_queue/logs/` のログは、完了から7日で自動削除される（`repro/` は対象外）。あなたの記録が `.training_queue/logs/` を参照していれば、ログを自分の保持先（自分の出力ディレクトリ）に SHA-256 付きでコピーし、記録の参照と読込をそちらに揃える。親が #2492 関連のログ101件を `.worktree-outputs/parent-2492-queue-log-archive/logs/` に複製済み（SHA256SUMS 付き）なので、元ログがすでに消えていたらそこから取ってよい。これから入れる job のログも、完了後に同じようにコピーする。
3. GPU は training queue 経由で、小さい job にまとめる。

## 終わり方
テストと pre-commit を通して push する。最終回答には、確かめたスレッド設定、入れた job、push した HEAD を書き、queue を待つ `PARENT_RESULT`（`job_ids` 付き）を返す。
