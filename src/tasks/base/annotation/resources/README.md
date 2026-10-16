# 工程の受付と資源予算

`ResourceManager` はcampaignをまたぐ予算scope、工程枠、上限付き待機列を管理する。
scopeは同じ一連の実験・追加学習で共用する。campaign IDを変えても使用済みGPU時間や
保存量はリセットしない。scopeとcampaignの対応は登録後に変更できない。

`create()` に合意した上限、計測root、開始前に使用した保存量/GPU時間、工程ごとの
待機数上限を渡す。productionの保存計測にはmain repository全体を必須とする。
Codexのsession/logが外部にある場合もrootとして含める。baselineとscopeは不変に保存し、
再起動時はその参照から開く。GPUの予約値は使用率ではなく、ジョブ実行時間の上限秒数。
複数GPUジョブの同時実行はそれぞれの経過秒を合算する。

`free_space_mounts` は、保存量の走査には加えず空き容量だけを確認するmount pointの一覧。
たとえばWSLのLinux領域を保持するWindows driveを `/mnt/d` として指定し、Linux内の
空きとhost側の実空きを両方確認する。`roots` と追加mountの空き容量の最小値を受付と
watchdogに使う。mount消失やsymlinkへの変更は計測エラーとし、親filesystemへ代替しない。
この設定もscopeとともに不変保存するため、既存scopeへの追加ではbaselineと累計を保持した
明示的なscope移行が必要。VHDX自体を保存量のrootへ加えて二重計上しない。

`submit()` は不変planと工程、PSSの予想ピーク、追加保存量、GPU予約時間を固定する。
同じrequest IDの内容変更や完了後の再実行は拒否する。明示的な再試行には新しいIDが必要。
待機列、同名工程、Codex合算の枠はcampaign横断で共有し、稼働中planのうち厳しい上限を
適用する。GPU論理枠の排他は別途共有training queueが所有する。

`submit(inputs=...)` は明示した入力参照とplan・scopeを、pending要求と同じtransactionで
保存台帳へ保護登録する。plan内の直接参照とscopeのbaselineも保護する。入力集合の変更は
同じrequest IDでは拒否する。以後の `submit()` / `lease()` は登録済み集合を再利用する。
受付時には試行用の保護をrunning状態と同時に登録し、lock解放と所有process消滅の両方を
確認した回収transactionで解除する。`finish()` の終了記録だけでは解除しない。
取消・失敗・正常終了後も要求の入力保護は残し、再開や後続工程への引継ぎ前に消さない。
この入力保護の退役、過去の未登録工程への遡及、実際のGC削除はまだ行わない。

`lease()` のFDを新しいsupervisorへ継承し、`attach()` で隔離process groupの所有者を登録する。
親が起動直後に停止しても、子が保持するkernel lockで二重受付を防ぐ。
終了時は `finish()` を記録し、lock解放と所有processの消滅を確認してから枠を回収する。
lockだけが消えた場合も、生きた子孫がいる間は枠を保持する。
終了記録のないGPU処理は予約全額を保守的に計上し、`reserved_upper_bound` と表示する。
正常終了は `measured_wall_seconds` として実行時間を計上する。

計測の定義:

- 保存量: `lstat` の `st_blocks * 512` をdevice/inodeで重複排除する。
  通常fileに加えdirectory/symlink自身の割当も数え、symlinkの参照先へは入らない。
  開始前inodeの割当を差し引き、正の増分を合算する。古い素材を削除しても、
  別の新規出力を追加した分は相殺しない。論理bytes・entry数・unique inode数も別に残す。
  baselineの元pathは保護する必要があり、移動・削除すると保守的に追加量へ数える。
- 保存走査のI/O: Linux mount情報で9p上と確認したroot（WSLのWindows driveなど）は、
  metadata取得とdirectory列挙を最大8並列・待機64件以内で行う。全entryを毎回照合し、
  symlink非追跡、inode重複排除、baseline anchor、permission/I/Oエラーの扱いは変えない。
  通常のLinux filesystemは従来の逐次走査を使う。
- PSS: 所有groupの全processと `observe()` したcontrollerをPID/start-timeで照合し、
  `/proc/PID/smaps_rollup` のPSSを重複なく合算する。RSSへの代替はしない。
  各groupは実測PSSに加え、予想ピークの未使用部分を予約する。
- I/O: 各processの `/proc/PID/io` の累計read/write bytes。時系列差分は同じPID/start-timeの
  間でのみ求める。終了直前の未観測I/Oを推定値で埋めない。
- VRAM/使用率: `nvidia-smi` のdevice全体と、報告された所有compute PIDの割当を区別する。
  WSL等で未対応/N/Aのときは `unavailable` と理由を残し、0使用とみなさない。

`sample()` は全項目を更新する。受付時はPSS・空き容量を再計測し、保存量は最長30秒の
cacheを使う。終了した工程の保存予約を解放するときは、cacheを無効にして再計測する。
controllerの `poll()` は共有した2秒間隔のPSS計測、30秒間隔の保存計測を用い、
保存計測日時も記録する。
supervisorは `watch()` でPSS・空き容量・経過GPU時間を確認する。こちらは別のlockで
2秒に1回共有し、全保存走査や `nvidia-smi` を呼ばない。保存走査中もtimeout判定を
進められる。`watchdog.jsonl` にPSS等を残し、保存量には最後の実測日時を付ける。
watchdogは保存予約を解放せず、admission用sampleを書き換えない。新規受付は全走査の
lockで待つため、終了工程の未計測出力を無視して次を起動しない。
baselineの検証済みhashはdevice/inode/size/mtime/ctime/mode/pathに結び付けて共有する。
どれかが変われば再hashし、mtimeを戻した同サイズの改変も拒否する。
上限到達は永続stopにし、**新規起動**を止める。空き容量不足や上限到達を無制限retryで
回避しない。原因解消後は理由を付けて `resume()` する。予約不足による待機と、
実測によるstopは区別して `status()` / `samples.jsonl` に残す。

再開では全項目を再計測し、上限を満たした場合にscope参照・理由・解除前のstopを含む
計測結果を `BudgetResumption` として不変保存する。その参照の `resource_resumptions` への
登録とstop解除を同じtransactionで確定する。保存やtransactionが失敗した場合はstopを維持し、
DBへ登録されていないファイルだけでは再開済みと判断しない。確定後のsample cache更新が
失敗しても、理由と計測根拠は同じregistryに残る。`resume()` と `resume-budget` CLIは
この参照を返す。scope・baseline・GPU累計は変更しない。

これはOSのメモリ・disk hard quotaではない。計測中の増加や見積もり超過は起こり得るため、
実測ピークと超過の有無を報告する。通常workloadの所有境界はtraining queueと同じ
process groupであり、`setsid` / daemon化で境界外へ出る子は対応しない。

production `WorkerPool` はscopeとPSS/保存予約を必須とし、supervisorへleaseを継承する。
[execution](../execution/README.md) がCPU/I/O工程とqueue内GPU工程へ同じ受付を適用する。
GPU処理にはtimeoutに加えて10秒以上の終了処理余裕を予約する。OSの停止遅延などを含め、
実測で予約を超過した場合も切り詰めず計上する。全工程の接続と実測は #1141 / #1143 で確認する。
資源テストのGPU予約はsleepだけの合成工程であり、CUDA実測を意味しない。
GPU累計時間の上限は正の秒数、または明示した `gpu_seconds: null`（上限なし）。
省略はエラー。上限なしでも工程ごとの予約、経過時間・使用済み累計の記録、
training queueの排他、保存量・空き容量・PSSの制約を維持する。

