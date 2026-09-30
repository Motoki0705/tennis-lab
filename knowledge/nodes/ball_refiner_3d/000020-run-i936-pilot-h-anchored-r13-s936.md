---
id: run-i936-pilot-h-anchored-r13-s936
type: run
task: ball_refiner_3d
sequence: 20
recorded_at: '2026-09-30'
title: 新bank・H固定の640ラリー生成をdetached起動
issue: 936
provider: codex
status: running
date: '2026-09-30'
config:
  plan: knowledge/runs/run-i936-anchored-dev-comparison-r12-s936/pilot-plan.json
  mode: pilot
  seed: 936
  counts: {train: 512, val: 64, test: 64}
  workers: 4
  native_threads: 1
  nice: 10
metrics: {}
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
