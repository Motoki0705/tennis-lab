---
id: run-i935-context-fullframe-pilot-r10-20260928
type: run
task: ball_refiner
sequence: 7
recorded_at: '2026-09-28'
title: JPEG文脈pilotが全561frameで成功し、観客混入・track増加・court欠損を監査
issue: 935
provider: codex
session: 01a0e84e-36f0-7203-bec2-b324e2db9aa0
date: '2026-09-28'
status: done
config:
  sources: [tracknet, meiji, chat_annotation]
  split: train
  clips: 3
  max_tracks: 64
  person_region_policy: full_frame
  pose_threshold: 0.15
  court_frames: [0]
metrics:
  completed_clips: 3
  completed_frames: 561
  observed_pose_crops: 3814
  valid_arm_slots: 14508
  saturated_arm_slots: 25
  total_dense_arm_slots: 94472
  clips_with_valid_court: 2
repro:
  commit: c90b6153b3588cc1a73c238e93ee4e7030dc4304
  branch: campaign930/i935-8-context-cache
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --signal=TERM --kill-after=15s 18m bash /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-8-context-cache/tests/benchmarks/ball_refiner_context_pilot.sh
    /home/kamimura/projects/tennis-lab /home/kamimura/projects/tennis-lab/data/ball_refiner/detector-ft-e13-trainval-r3-20260928
    /home/kamimura/projects/tennis-lab/data/ball_refiner/context-fullframe-pilot-r10-20260928
    /home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/context/i935-context-fullframe-pilot-r10-20260928
artifacts:
  run_dir: knowledge/runs/run-i935-context-fullframe-pilot-r10-20260928
  log: knowledge/runs/run-i935-context-fullframe-pilot-r10-20260928/queue.log
parents:
- run-i935-evidence-ft-e13-trainval-r3-20260928
relations:
- to: run-i935-context-fullframe-pilot-r9-20260928
  rel: retries
papers: []
tags: [context, pilot, dino, vitpose, full-frame, quality-audit]
---

## 結果と証拠

run専用に再ビルドしたDINO拡張で、前回停止した3source pilotが完走した。
queue job `1790604762249484170_650104_i935-context-fullframe-pilot-r10-20260928` はdone。
全561frameの人物検出・追跡、実観測boxの3,814 cropへのViTPose、
各clipのframe 0 court探索が完了した。生成cacheのmanifest SHA-256は
`d20516d29f6b80272dc7c5dc8a5900f8064b46dbe8067d7833469a7475b0e742`。
保存後の別Pythonプロセスが全3 NPZのchecksum・元frame/PTS・JPEG hash・
stored/source座標変換を検証した。CPU監査でも同じNPZを全読込し、
画像の前後hashとmanifest不変を確認した。

- [生成・読込の検証](../../runs/run-i935-context-fullframe-pilot-r10-20260928/context-verification.json)
- [CPU監査・全frame統計](../../runs/run-i935-context-fullframe-pilot-r10-20260928/audit.json)
- [生成manifest・資産/code hash](../../runs/run-i935-context-fullframe-pilot-r10-20260928/context-manifest.json)
- [scene設定](../../runs/run-i935-context-fullframe-pilot-r10-20260928/scene-context.yaml) /
  [事前検査](../../runs/run-i935-context-fullframe-pilot-r10-20260928/preflight.json) /
  [build設定](../../runs/run-i935-context-fullframe-pilot-r10-20260928/build.json)

| trainのpilot clip | frame | 累計track / 最大同時観測track | pose crop | 有効な肘/手首slot | court |
|---|---:|---:|---:|---:|---:|
| TrackNet game5/Clip14 | 35 | 19 / 18 | 576 | 2,270 | 14/14 |
| Meiji video_002/clip_011/cam0 | 151 | 3 / 3 | 450 | 1,604 | 14/14 |
| chat -6UwVW0DeO4…496–821 | 375 | 60 / 13 | 2,788 | 10,634 | 0/14 |

score >= 0.15の肘/手首が一つ以上あるframeは561/561、画像外の有効slotは0。
これは**いずれかの人物**が観測された比率であり、プレー中の人物のrecallではない。
scoreの1超は25 / 94,472 dense slotで、全てchat（最大1.024473）。
分母は全frame×累計track×4で、未観測のゼロslotも含む。
観測crop×4を分母にすると25 / 15,256。有効slotは14,508である。
値は非負heatmap peakを1へ飽和したもので、確率の較正ではない。

## 画像監査で確認した限界

ラベルを見ずに各clipの先頭・1/4・中央・3/4・末尾の5frameを固定して重ねた。
黄色は平滑化・拡張後のcropとraw track ID、緑は肘/手首と肩からの腕、
紫はframe 0のcourt KP14。対象人物の手首GTによる精度測定ではない。

- [TrackNetの5frame](../../runs/run-i935-context-fullframe-pilot-r10-20260928/clip-00062-context.jpg):
  手前サーバーと奥のプレーヤーの腕を捉える一方、観客・コート周辺の人物も多数含む。
  courtは表示された主コートの線に概ね沿うが、近側の外側点などに目視のずれがあり、GT精度は未測定。
- [Meijiの5frame](../../runs/run-i935-context-fullframe-pilot-r10-20260928/clip-00227-context.jpg):
  手前と奥の対象に加え左側の隣接コート人物を含む。遠方人物は小さく、
  14点が有効でも対象コート・各点の位置の正しさを保証しない。別GTとの照合は未実施。
- [chatの5frame](../../runs/run-i935-context-fullframe-pilot-r10-20260928/clip-00266-context.jpg):
  プレー中の人物だけでなく周辺スタッフ・観客にもposeがある。frame間で視点・画角が変化し、
  trackが断片化して累計60に達した。frame 0 court探索は支持領域なしとして明示的に欠損を保存した。
  未実行ではなく、別推定への切替もしていない。このclipはcourt全欠損なので
  不適切な固定courtが後続frameに残る問題を直接測ったわけではない。

全画面推論では観客混入を避けられず、長いclipの累計64track上限も危うい。
一方、人物集合を学習へ渡す設計であり、観客を含むことだけを根拠に
品質GTなしの人物選別・切捨てへ切り替えるべきではない。
元cacheを保持し、生成に成功したsubsetだけでfull効果を主張しない。
可変視点に対するframe 0 court priorの弱点も、文脈なし/poseなし比較と併せて評価する。

## 速度と次の実験

実測の検出+追跡 / pose / courtは、TrackNet 15.76 / 39.14 / 30.29秒、
Meiji 48.00 / 31.12 / 12.60秒、chat 116.70 / 156.71 / 20.12秒。
合計470.45秒には各clipのモデル読込・decode等が含まれ、
拡張build・外側のhash/読込検証は含まれない。全sourceから最短のtrain clipを選んだpilotで、
代表的なthroughputや全体runtimeの推定には不十分である。
単純なsource別frame比例では全329clip・145,767frameに約37時間となるが、
起動費用・人数・camera差を無視した参考外挿にすぎない。

次は全sourceで同じモデル/code/RGB条件を固定し、clip単位で独立に失敗・再実行できる
分割生成と、重複/未生成/異なるidentityを拒否する全被覆の統合契約を用意する。
累計track上限はモデルの同時人物数と分けて設計し、超過clipを黙って除外しない。
fullと各ablationは共通のtrain/validation母数で学習・選択する。
今回の成功は文脈生成と保存契約の証拠で、文脈による精度改善・学習戦略の合意ではない。
RGB遮蔽対照・amodal疑似教師品質・較正fit・最終Meiji testは未完了で、deploy判断は変えない。

## 再現

生成元commitはc90b6153。queueのrun.json / repro.sh / clean statusを内容不変で保存し、
追跡済みのbenchmarkからDINO拡張を再ビルドできる。旧runとは異なるbinaryでCUDAを実行した。
元出力は上書き不可なので、再生成時は新しいcache/report pathを明示する。
全3 NPZを本bundleのclips/へ保持し、raw JPEG・checkpoint・binaryは複製しない。
保存後CPU監査は[audit_context.py](../../runs/run-i935-context-fullframe-pilot-r10-20260928/audit_context.py)へ
絶対pathの `--store`・`--evidence`・`--context`・新規 `--report` を渡す。
元JPEGや重みの入手性までは保証しない。学習ではないためTensorBoard曲線はない。
