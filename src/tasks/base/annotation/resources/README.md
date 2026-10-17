# 工程の受付と資源予算

`ResourceManager` はcampaignをまたぐ予算scope、工程枠、上限付き待機列を管理する。
scopeは同じ一連の実験・追加学習で共用する。campaign IDを変えても使用済みGPU時間や
保存量はリセットしない。scopeとcampaignの対応は登録後に変更できない。

`create()` に合意した上限、計測root、開始前に使用した保存量/GPU時間、工程ごとの
待機数上限を渡す。productionの保存計測にはmain repository全体を必須とする。
Codexのsession/logが外部にある場合もrootとして含める。baselineとscopeは不変に保存し、
再起動時はその参照から開く。GPUの予約値は使用率ではなく、ジョブ実行時間の上限秒数。
複数GPUジョブの同時実行はそれぞれの経過秒を合算する。

`submit()` は不変planと工程、PSSの予想ピーク、追加保存量、GPU予約時間を固定する。
同じrequest IDの内容変更や完了後の再実行は拒否する。明示的な再試行には新しいIDが必要。
待機列、同名工程、Codex合算の枠はcampaign横断で共有し、稼働中planのうち厳しい上限を
適用する。GPU論理枠の排他は別途共有training queueが所有する。

`lease()` のFDを新しいsupervisorへ継承し、`attach()` で隔離process groupの所有者を登録する。
親が起動直後に停止しても、子が保持するkernel lockで二重受付を防ぐ。
終了時は `finish()` を記録し、lock解放と所有processの消滅を確認してから枠を回収する。
lockだけが消えた場合も、生きた子孫がいる間は枠を保持する。
終了記録のないGPU処理は予約全額を保守的に計上し、`reserved_upper_bound` と表示する。
正常終了は `measured_wall_seconds` として実行時間を計上する。
回収はlease取得後に最新の終了記録を読み、同じtransactionで状態を更新する。
回収待ちの間に終了しても、記録済みの実測時間・計上種別・終了理由を保持する。

計測の定義:

- 保存量: `lstat` の `st_blocks * 512` をdevice/inodeで重複排除する。
  通常fileに加えdirectory/symlink自身の割当も数え、symlinkの参照先へは入らない。
  開始前inodeの割当を差し引き、正の増分を合算する。古い素材を削除しても、
  別の新規出力を追加した分は相殺しない。論理bytes・entry数・unique inode数も別に残す。
  baselineの元pathは保護する必要があり、移動・削除すると保守的に追加量へ数える。
- PSS: 所有groupの全processと `observe()` したcontrollerをPID/start-timeで照合し、
  `/proc/PID/smaps_rollup` のPSSを重複なく合算する。RSSへの代替はしない。
  各groupは実測PSSに加え、予想ピークの未使用部分を予約する。
- I/O: 各processの `/proc/PID/io` の累計read/write bytes。時系列差分は同じPID/start-timeの
  間でのみ求める。終了直前の未観測I/Oを推定値で埋めない。
- VRAM/使用率: `nvidia-smi` のdevice全体と、報告された所有compute PIDの割当を区別する。
  WSL等で未対応/N/Aのときは `unavailable` と理由を残し、0使用とみなさない。

`sample()` は全項目を更新する。受付時はPSS・空き容量を再計測し、保存量は最長30秒の
cacheを使う。終了した工程の保存予約を解放するときは、cacheを無効にして再計測する。
稼働中の `poll()` は全workerで共有した2秒間隔のPSS計測、30秒間隔の保存計測を用い、
保存計測日時も記録する。
上限到達は永続stopにし、**新規起動**を止める。空き容量不足や上限到達を無制限retryで
回避しない。原因解消後は理由を付けて `resume()` する。予約不足による待機と、
実測によるstopは区別して `status()` / `samples.jsonl` に残す。

これはOSのメモリ・disk hard quotaではない。計測中の増加や見積もり超過は起こり得るため、
実測ピークと超過の有無を報告する。通常workloadの所有境界はtraining queueと同じ
process groupであり、`setsid` / daemon化で境界外へ出る子は対応しない。

production `WorkerPool` はscopeとPSS/保存予約を必須とし、supervisorへleaseを継承する。
取得・画像生成・公開のCPU/I/O工程、推論・追跡・pose・学習のqueue内受付は同じAPIを
composition rootから利用する（全工程の接続と実測は #1141 / #1143 で確認する）。
GPU処理は予約内で終了するtimeoutと子孫終了確認をqueue clientで設定する必要がある。
資源テストのGPU予約はsleepだけの合成工程であり、CUDA実測を意味しない。
