---
id: group-i935-context-shards-r12-probe
type: group
task: ball_refiner
sequence: 11
recorded_at: '2026-09-29'
title: 全329clip固定計画の最長3source probeと生成継続判断
members:
- run-i935-context-shard-00059-tracknet-r12-20260928
- run-i935-context-shard-00089-meiji-r12-20260928
- run-i935-context-shard-00201-chat-r12-20260928
parents:
- run-i935-context-fullframe-pilot-r10-20260928
papers: []
tags:
- context
- shards
- quality-audit
issue: 935
provider: codex
status: done
date: '2026-09-29'
session: 01a0e8af-3304-7740-a0ef-645d6f9a9dc6
config:
  planned_clips: 329
  planned_frames: 145767
  selection: longest_clip_per_source
  max_tracks: 1024
metrics:
  completed_clips: 3
  completed_frames: 3156
  remaining_clips: 326
  remaining_frames: 142611
---

## 何が確認できたか

全329clip / 145,767frameを固定した計画の下で、sourceごとの最長clipを1件ずつ生成した。
3jobはすべてdoneとなり、全3,156frameの人物検出→BoT-SORT→実観測cropのViTPoseと
各frame 0 court探索、別Pythonプロセスの保存後検証が完了した。
run13のCPU再監査もNPZ・実JPEG hash・frame/PTS・座標変換・manifest不変を確認した。

| probe | frame | 累計track / 最大同時観測 | pose crop | stage時間合計 | queue壁時計 | court |
|---|---:|---:|---:|---:|---:|---:|
| [TrackNet game7/Clip4](000008-run-i935-context-shard-00059-tracknet-r12-20260928.md) | 901 | 31 / 15 | 11,137 | 927.25秒 | 967秒 | 14/14 |
| [Meiji video_000/clip_002/cam0](000009-run-i935-context-shard-00089-meiji-r12-20260928.md) | 1,355 | 22 / 6 | 6,389 | 787.07秒 | 829秒 | 14/14 |
| [chat 1DbPtcuuS-Q…773–1553](000010-run-i935-context-shard-00201-chat-r12-20260928.md) | 900 | 43 / 12 | 8,013 | 745.08秒 | 784秒 | 0/14 |

旧pilotは各sourceの最短train clipであり、今回とclip・生成code・binary・track上限が違う。
今回のprobeは最大frame数だけで選び、ラベル・品質・学習指標で選んでいない。
Meijiはvalidation clipの文脈監査で、ballの精度によるcheckpoint/ハイパーパラメータ選択をしていない。
累計上限1024では切捨て・track ID再利用なしで完了したが、この3件は旧64上限にも収まり、
1024の必要性や全clipでの容量上限を実証したわけではない。

## 品質と判断の限界

各5frameの画像では観客・周辺人物・隣接courtの人物を含む。Meijiのcourtは14点が有効でも
奥側へ偏り、手前側の線とのずれ・対象コートの曖昧さが見られる。chatは支持領域なしとして
実行済みcourt欠損を保存した。人物/コートのGTによる精度は測定していない。
全frameに有効な肘/手首があることは、プレー中の人物を正しく捉えた保証ではない。
TrackNetの画像外関節も有限値なら保持し、都合のよいsubsetや事後maskへ変えていない。

[run13の暫定判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5873558234)に従い、
元cache・品質診断を保持したまま同一条件で残り326clipを生成する。
文脈の有効性は全被覆の同一train/validation母数・seed・更新数のfull/poseなし/文脈のみと
pose摂動で検証する。成功subsetだけをfullと呼ばず、court/pose有無でも層別する。
この段階で学習設計へのユーザー合意、較正の成功、精度改善、deploy採用は主張しない。

## 生成を継続する条件

[固定plan](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/plan.json)のSHA-256は
`1730b845df7348a9f800575d8fbf0442b04324fcfa27d10ee87e005081af3486`。
共有scene・checkpoint・生成src全体・DINO binaryを変更しない。
[継続前のCPU照合](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/continuation-preflight.json)でも
現在のmodel/code/binary identityとplanの一致、残り326clipの新規出力先を確認した。
残りclipの実JPEG再hashは各生成jobと全体統合で行う。
1clip全frameを1jobで処理し、trackingを途中分割しない。resource=all、各jobは引き続き18分上限とする。
今回の最長3件はその上限内だが、人数の多い短clipの処理時間を上から抑える証拠ではない。
timeout/例外時も元出力を保持し、原因を回収して別attempt pathで再実行する。
完了後に採用する成功pathを明示し、全329clipの被覆とidentityを統合CLIで検証する。

元の各queue job・run.json・clean status・logとNPZを保存した。共通plan・scene・build/preflightは
最初のprobe bundleを正本とする。元の `repro.sh` は内容不変の `captured-repro.sh` として保持し、
未保存のbinary・重み・JPEGと絶対pathに依存するGPU commandを、配布可能なreplayとは呼ばない。
新binaryで再生成する際には新しいidentity/planが必要である。
GPU生成中は専用worktreeを凍結し、学習runnerの開発は後続worktreeで行う。
較正fit・dense detectorの密度比較・RGB遮蔽対照・pseudo-amodal品質・最終Meiji testは引き続き未完了。
