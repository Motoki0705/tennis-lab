---
title: "WSL2 のホストRAM不足では、学習プロセスがtracebackなしで黙って殺される"
summary: "WSL2のRAM不足で学習や親のbackground workerがtracebackなしで止まった。WSL内の上限と同時に動くプロセスを調べる。"
applies_to: all
source: "DINOv3 LoRA-SSL 学習（2026-06-30）、court 学習中のWSL2再起動（2026-07-05）、#2010 の親セッションで worker が止められた件と人間による上限変更（2026-11-15 の会話）"
created: 2026-06-30
last_verified: 2026-11-20
evidence: "free -g で2026-11-20も total は39。Windows側の設定は2026-11-15の人間による変更（memory=40GB）。停止は過去の観察で、再現していない。"
citations: []
---

画像約30万枚の DINOv3 SSL 学習が、ログの途中で何も出さずに止まった（3回）。VRAMは約4.8GBで余裕があった。停止位置は `num_workers` に比例して前後し、新しいプロセスで再開するとリセットされた。DataLoader worker の RSS が少しずつ増え（大きなパスのタプルの copy-on-write）、WSL2のホストRAMの上限に達したと考えている。WSL2そのものが再起動することもある（2026-07-05）。WSL2 の上限は `.wslconfig` の `memory` で決まり、2026-11-15 に人間が24GBから40GBへ上げた（物理64GB）。Windows側に空きがあっても、WSL内からは上限までしか使えない。

2026-11-14 には上限24GBのとき、ボール検出の学習（約7GB）と、親4人分の Codex worker が走らせる pytest の並列プロセス（各0.6GB前後）が重なった。このとき Claude Code がWSL内のメモリ不足を検知して、待機中のbackground shell（親の `run-worker`）を止め、配下の共通ランナーと Codex も終了した。

**使い方:** tracebackなしで止まったら、まずWSLのメモリ上限を疑う（`free -g` と `.wslconfig` を見る）。親が worker を並列に起動するときは、学習ジョブや他の親の worker を含めた合計を見積もる。`num_workers` を減らす、checkpoint の間隔を詰める、自動で再開する watchdog を付ける、GPU学習中に重いCPUジョブを並行させない、を試す。
