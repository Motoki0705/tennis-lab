# GPU停止の切り分け（人間の許可あり、2026-11-19 14:57 JST）

#1757の逐次probeが2回（13:36と14:30）、DINO fp16の最初の推論が終わった直後に止まりました。nvidia-smiが10秒以内に応答しなくなっています。親が調べた結果は次のとおりです。

## 親の調査結果
- 同じ止まり方が、過去に#1751のa3（11/17 21:13）とa4（11/18 01:40）でも起きていた。a4はDINOのdeformable transformerの直後。この2回は、Windowsのイベントログにnvlddmkm ID 153が同じ秒に記録されている。
- 一方、同じDINO拡張・同じfp16で、#1751 a5、#1752の3clip、#1753は完走している。
- 止まったとき、プロセスはSIGUSR1にも応答しなかった。WSLのカーネルには、`dxgvmb_send_sync_msg: wait_for_completion failed` が出ている。
- VRAMは3.5GB、温度・電力も平常。DINO拡張はsm_120向けに正しくビルドされている。driver 621.24の前後どちらでも起きている。

## 確かめたいこと
人間の許可を得て、次の切り分けを行う。

1. **条件A**: DINOだけを、clip_027/cam0の1カメラで、fp16、標準の設定（#1757のprobeと同じ重み・拡張・入力）で動かす。最低でも、probeが止まった位置（最初の推論から数十frame先）を十分に越えるframe数にする。
   - GPU監視でnvidia-smiを呼ばない。メモリはプロセスの中で、`torch.cuda.memory_allocated` / `max_memory_allocated` を取る。
   - 停止の検知は、プロセスの中のheartbeatファイルと、外部の時間制限で行う。たとえば、heartbeatが60秒止まったら、記録してから停止させる。
2. **条件B**: 条件Aが最後まで止まらなかった場合だけ、同じ実行に、別プロセスから1秒ごとにnvidia-smiを呼ぶ監視（#1757のhealth_guardと同じ問い合わせ）を足して、もう一度実行する。
3. どちらかで止まったら、すぐに中断して時刻を記録する。再試行や次の条件には進まない。記録するものは次のとおり。
   - heartbeatの最後の位置とframe番号
   - `journalctl -k` の該当時刻のdxgの行
   - Windowsのイベントログ（nvlddmkm）。`powershell.exe Get-WinEvent` で読む。

## 実行の条件
- GPUは共有training queue経由で使う。`TRAINING_QUEUE_DIR` はrepo rootの `.training_queue/`。
- CPU負荷の影響を除くため、各条件の開始前に、1分のload averageと、他のworktreeのpytestの数を記録する。
  - 開始の条件は、load averageが2未満で、pytestが動いていないこと。
  - 最大30分待っても条件を満たさない場合は、その値を記録して開始してよい。
- 実行のスクリプトは `knowledge/runs/` の下に置き、#1757のbranchにcommitする。結果は、knowledgeのrun記録（failedでもよい）として残す。
- 学習でも、パイプラインの改変でもない。モデルの精度・batch・推論の方式は変えない。

## 最後に報告すること
- 各条件の結果（止まったか、何frame目か、所要時間）
- 記録したログの場所
- 親が次に判断するための材料（どちらの仮説が強くなったか）
