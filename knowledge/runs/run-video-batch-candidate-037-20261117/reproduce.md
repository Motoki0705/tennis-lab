このrunはclip_037の単一cold到達確認で、player_associationで失敗した記録である。
全段baseline・warm・完成prefix・全cache hitを測った記録ではない。
測定後のrebaseで原commitが失われないよう、以下をoriginのannotated tagへ固定した。

- 037: `evidence/i1748/clip-037-20261117` → `701165bd2deeddc20d4773b3dbacb6890f8c15ae`
- 旧035: `evidence/i1748/clip-035-20261116` → `cf4a1b0459588cf77f5ed81af9b69751c5b65ec7`

037のmain基点は486aa1ab09be857909d0fcf8c2b616c75a53f5b0。
sourceとdriverは実行前後で固定されていた（measured-source-after-run.json）。
報告用branchのrebase後のcommitでGPUを測ったという意味ではない。
queue/repro.sh・run.jsonは元のコマンド・session・source commitをそのまま保持している。

再実行するagentはtagをfetchし、AGENTS.mdとstorage-tier手順に従った別worktreeで
このtagをcheckoutする。benchmark.pyのfreezeは既存の完了済み037をoutputsへコピーする。
input-manifest.jsonのSHA-256
`497f75bf338ec5a918f9011cea789b7375bd184ce9407a7b6583126c24194002`と各動画/clip.jsonの
内容を照合する。元datasetへ書かない。既存の実測outputは上書きしない。

固定条件は037の4camera、292 frame、既定i935、片側1人、sampling cap1024、
1回のcold component cache、初回CUDA/model load込み。Torch/OpenCVのCPU importと入力検証・
コピーは窓外。OS page cacheは破棄しない。実行中に別processで元3clipのCPU診断replayを行った。
この条件差もあるため、035との速度比や最終baselineとは扱わない。

DINO/dinov3 submoduleはenvironment-after-run.jsonのgitlinkを用い、GVHMRはvendored
実装なので未初期化のthird_party/GVHMRを新しく用意しなかった。
既存のDINO拡張binaryの場所・SHAは同JSON。original035のrestored-artifacts.jsonとも一致する。
これは実行後に読んだ環境情報で、環境の変更を実測で否定したという記録ではない。

新しい出力先で再現するときの入口は次の通り。GPU runは必ず共有queueへ登録し、
再実行するagent自身のsession/idempotency keyを使う。下記は手順であり、失敗後に追加GPU runは行っていない。

```bash
git fetch origin tag evidence/i1748/clip-037-20261117
# このtagをcheckoutした専用worktreeと共有入力/venvを準備後、そのrepo rootで実行する。
.venv/bin/python knowledge/runs/run-video-batch-candidate-037-20261117/benchmark.py \
  freeze --root outputs/tennis_scene/evaluate/c1-037-replay
```

queueへ送るコマンドはqueue/run.jsonのcommandを基に、出力先だけ上記の新rootへ変え、
`--expect-commit 701165bd2deeddc20d4773b3dbacb6890f8c15ae --allocator-limit-gib 10`
を保つ。`TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue`、
`--resource all --provider codex --issue 1748`、現在のsessionを指定する。
最初の例外で停止し、人数・閾値・camera・入力を変更して回避しない。

GPUを使わない証拠の確認はmetrics.json・receipts・lifecycle・queue/completion.jsonから行える。
raw recorderはmetrics.jsonにあるoutputsのpath・サイズ・SHAで固定した。
元3clipのCPU後処理replayはcourt-default-replay/replay_original_inputs.py。
保存済みraw/LINE/RGBの場所とhashは同ディレクトリのinputs-and-source.jsonにある。
元の#2190 replay関数を呼ぶが、出力先だけ専用outputsへ変更し、保存済み入力を再生成しない。
