---
id: run-i936-pilot-h-anchored-r13-s936
type: run
task: ball_refiner_3d
sequence: 20
recorded_at: '2026-09-30'
title: 新bank・H固定の640ラリー生成完了と全保存hash監査
issue: 936
provider: codex
status: done
date: '2026-09-30'
config:
  plan: knowledge/runs/run-i936-anchored-dev-comparison-r12-s936/pilot-plan.json
  mode: pilot
  seed: 936
  counts: {train: 512, val: 64, test: 64}
  workers: 4
  native_threads: 1
  nice: 10
metrics: {rallies: 640, frames: 272986, generation_seconds: 22372.766095253988, dataset_bytes: 1431545595}
repro:
  commit: f8459a62c55776780c920a8c842c3fde977d6d23
  branch: campaign930/i936-2-synthetic-diffusion
artifacts:
  run_dir: knowledge/runs/run-i936-pilot-h-anchored-r13-s936
  output_dir: /home/kamimura/projects/tennis-lab/data/ball_refiner/i936-pilot-h-anchored-s936
  log: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/i936-pilot-h-anchored-r13-s936.log
parents:
- run-i936-anchored-dev-comparison-r12-s936
relations: []
papers: []
tags: []
---

run 13 directiveが640生成を明示的に許可したため、run 12の計画を2026-09-30 20:59:53 JSTに起動した。
[起動台帳](../../runs/run-i936-pilot-h-anchored-r13-s936/launch.json)がPID、実行コマンド、
新出力・log・manifestの絶対path、全入力hash、費用見積の正本。
PID/sessionは4044520、nice10、標準入出力を切り離した独立sessionであり、Codex終了後も継続する。
generatorの4worker以外のCPU作業は1processに限定する。

run 12の展開設定と完全一致し、seed936・H・K4/全125成分・512/64/64を維持した。
新bank SHAは0697fe921daf79c7960d616ed0437efecd3b7195f3f8a7dd6188048dd8ecc858、
report SHAは2c9cd7ba9a1d63addeaecb808441fbae0447ad21e4a3a279ed1694fcbf67ae01。
36入力のhashを起動前後に検証した。[開始manifest](../../runs/run-i936-pilot-h-anchored-r13-s936/initial-manifest.json)は
runningの証拠であり、完了・品質の検証結果ではない。生成sourceと入力は終了まで変更しない。
旧#959 controlは保持する。testは生成のみで品質採点・モデル選択に使わない。

見積8.1〜10時間、RAM4〜6GB、disk2GBはrun 12の96件からの外挿で、今回の実測値ではない。
起動前のMemAvailableは18.13GB。三角測量失敗によるseed変更、成分除外、自動retryを認めない。
quota休止後にmanifestのcomplete/failed、全640件のhash、入力hash不変、実測資源を回収する。
完了まで待機せず、この時点ではAcceptance第2項も未完了。CPU生成のため学習曲線/TensorBoardはない。


## Run 14の途中監査

[status-run14.json](../../runs/run-i936-pilot-h-anchored-r13-s936/status-run14.json)に観測時刻と途中manifestのhashを保存した。
453/640成功、失敗0、PID4044520稼働中。完了済み全NPZのhash/bytesとJSON一致、
全36入力hash不変、run12のexpanded planとの完全一致を確認した。既存devとの共通train/val 64件もNPZ hash一致。
これは完了監査ではない。test配列は開かず、失敗時の再生成もしない。run14の640学習は
complete・640件・失敗0・全hash・plan一致をGPU前preflightで要求し、未完了なら即停止する。

## Run 15: 全640件の完成監査

[final-audit/audit.json](../../runs/run-i936-pilot-h-anchored-r13-s936/final-audit/audit.json)が全NPZ/JSON SHA・生成入力・資源・固定val preflightの正本。
[最終manifest](../../runs/run-i936-pilot-h-anchored-r13-s936/final-audit/generation-manifest.json)と生成logも保存した。
640/640成功（train512/val64/test64）、failures/failed artifactsとも0、logも重複なし640件の完了を確認した。
全272,986frameは60000/1001Hz、K4/全125成分、固定H。全frameの積分収束は未評価であり、成功を収束済みとは解釈しない。
全640 NPZと1,281 JSON（各ラリーmetadata・progress、manifest）をhash化し、NPZ SHA/bytes・metadataとmanifestの完全一致、全seed/geometry/component数を検証した。
run12の展開planと完全一致し、run13起動時の全36入力hashも不変。元80train+valの全NPZはbyte一致。
旧#959 controlもmanifestと全96NPZのhash不変。test・追加48valの配列は開かず、全件のmetadataとfile hashだけを監査した。

生成時間はmanifestのperf_counterで22,372.766秒（6.2147時間）、全ラリー処理の合計89,226.240秒。
simulation合計1,595.263秒、triangulation合計87,486.195秒。
最大worker単体RSSは903,860,224 bytes（0.904GB）。4worker同時のRSS合計と親processのpeakは記録されておらず、実測した総RAM peakとは称さない。
起動ISO時刻から最終manifest mtimeまでは21,304.106秒（5.9178時間）で、単調時計との不一致は未解明のまま両方を残す。
NPZ合計1,417,904,929 bytes、JSON/progressを含むdataset全体1,431,545,595 bytes。2GB予約以内で、今回新たに生成したのではなくrun13からの累積である。

再現はrepo rootで`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python knowledge/runs/run-i936-pilot-h-anchored-r13-s936/audit_final.py`。
既存final-auditは上書きしない。全配列読込0、GPU使用0。train/valの品質や最終較正を新たに採点した監査ではない。
元の(a) planに対して`preflight()`をそのまま実行して合格し、64固定ファイル・元16val/6,383frame・run13からtrain数だけの差分を確認した。
本番job用のpreflight/output先には書かず、再投入時にも同じgateを再実行させる。
Acceptance第2項の生成部分は完成したが、新#935の最終独立較正は未完。第3項の優位と第4項の統合も未完である。
