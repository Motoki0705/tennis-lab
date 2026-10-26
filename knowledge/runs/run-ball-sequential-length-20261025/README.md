# #1514の逐次学習

研究判断と実測結果の正本は[実験ノード](../../nodes/ball_detection/000068-run-ball-sequential-length-20261025.md)。
このディレクトリには実行手順を置く。既存の事前学習重みは利用しない。

実行は専用worktreeから、元repoの共有training queueを通す。
720p、5ch、BF16、compile有効、checkpointing無効、時間分割なし、seed 42。
共通DPT T32の10k更新からDPT T64の6kとquery T32→T64の各6kへ分岐する。
T128はユーザーの指示で除外した。

## 入力と環境

入力データは元repoの
`data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1`。
ダウンロードの検証記録は同repoの
`outputs/ball_detection/precompute/sequential_length/drive-download-20261025/`に置く。
`download_inputs.py`は既存のDrive mountから全ファイルを取得し、サイズと独立したSHA-256を検証する。
元の配布データ、注釈、splitを変更しない。

実測環境はPython 3.11.17、Torch 2.14.1+cu130、torchvision 0.29.1+cu130、NVIDIA L4。
`dual_sha256`の独立providerが利用可能であることを確認する。
Drive保存には認証済みmountを指す明示的なrclone aliasを使用した。
認証・保存の方針は[tennis-drive skill](../../../.agents/skills/tennis-drive/SKILL.md)を参照。
設定ファイルとcredentialはこのbundleへ含めない。

## 実行

以下のworkspace path、run ID、rclone configは今回の実行で指定した値。
GPUコマンドは[training-queue skill](../../../.agents/skills/training-queue/SKILL.md)に従って登録する。
ベンチマークと本学習を同時に実行しない。

```bash
.venv/bin/python -B knowledge/runs/run-ball-sequential-length-20261025/run.py prepare \
  --workspace-root /content/tennis-lab

PATH=/tmp/ball-tools:$PATH .venv/bin/python -B knowledge/runs/run-ball-sequential-length-20261025/run.py benchmark \
  --workspace-root /content/tennis-lab --run-id benchmark-20261025-a \
  --rclone-config /tmp/ball-drive-mount.conf

.venv/bin/python -B knowledge/runs/run-ball-sequential-length-20261025/query_gpu_smoke.py

PATH=/tmp/ball-tools:$PATH .venv/bin/python -B knowledge/runs/run-ball-sequential-length-20261025/run.py train \
  --workspace-root /content/tennis-lab --run-id s42-20261025-main \
  --rclone-config /tmp/ball-drive-mount.conf \
  --benchmark /content/tennis-lab/outputs/ball_detection/train/sequential_length/benchmark-20261025-a/benchmark-complete.json
```

benchmarkは本番と同じデータ・拡張・128 frame/updateでBS1/BS2を各30更新し、
全validationの所要時間とDrive保存時間を測る。最速の実測batchを本学習で採用する。
30更新の重みは速度検証専用で、本学習は新しく初期化する。
`query_gpu_smoke.py`は合成入力を使うT64のGPU診断であり、精度・収束の実験ではない。
診断では凍結/解除の2グラフを1更新ずつ確認し、本学習の凍結は各事後段階600更新を保つ。

## 状態確認と再開

本学習の正本はworkspaceの
`outputs/ball_detection/train/sequential_length/s42-20261025-main/`。
`plan.json`が全段階の計画、各段階の`train.jsonl`が更新ログ、`latest.json`が再開checkpointを示す。
各段階の`COMPLETED.json`とcampaignの`COMPLETED.json`を区別する。
ディレクトリや途中checkpointの存在だけで完了と判断しない。

500更新ごとに既存`RcloneSnapshotStore`でcheckpoint、weight、設定、manifest、ログを
Driveの同じoutputs相対pathの世代別snapshotへ保存する。
receiptはcampaign直下の`persistence/<stage>/<generation>.json`にあり、Driveへも転送する。
過去世代を自動削除しない。保存の失敗は学習失敗として扱う。

同じVMで中断した場合は元のcommit、入力、prepared manifest、benchmarkを保持し、
上記trainコマンドへ`--resume`を付けてqueueへ新しいattemptとして登録する。
残っているactive jobへ重複して登録しない。
VMが消失した場合は、Driveのreceiptで指定されたgenerationと`manifest_sha256`を使い、
`RcloneSnapshotStore.restore()`で元の段階pathへ検証付きで復元する。
ローカルpathが既存の場合、restoreは上書きしない。
元の`plan.json`、benchmark、prepared manifest、入力データも同じpathへ戻す。
コードや計画が変わった実行を同じcampaignとして黙って再開しない。

## VM消失後の復旧（2026-10-26）

データ取得は上の`download_inputs.py`を最初に実行する。Python 3.11.17の
`.venv/bin/python`を使い、`uv sync --locked`で保存時と同じTorch／torchvisionを
準備する。新しい学習も再開も、全2,925ファイルのサイズとSHA-256の検証が
完了していない場合は開始を拒否する。

Drive mountと、そのmountを参照する明示的な`mounted_drive` rclone aliasを
用意したうえで、次を実行する。mountからローカルSSDへデータを取得し、
学習中はローカルの入力を使う。

```bash
.venv/bin/python -B knowledge/runs/run-ball-sequential-length-20261025/recover.py \
  --workspace-root /content/tennis-lab \
  --drive-root /content/drive/MyDrive/tennis_lab \
  --rclone-config /tmp/ball-drive-mount.conf

# データ取得の完了後、共有training queueから起動する。
.venv/bin/python -B knowledge/runs/run-ball-sequential-length-20261025/run.py train \
  --workspace-root /content/tennis-lab --run-id s42-20261025-main \
  --rclone-config /tmp/ball-drive-mount.conf --resume
```

`recover.py`は保存receiptがある最新更新を選び、同じ更新では完了世代を優先する。
receipt・marker・全payloadのhashを検証し、common DPT T32の10,000更新、
DPT T64の6,000更新、query T32の3,000更新を復元できる。
同じ更新に複数の候補がある場合や、最新receiptが破損している場合は停止する。
再実行時に異なるローカル状態を上書きしない。prepared manifestの内容と
データの絶対pathも、checkpointに保存されたrecipeと照合する。

旧実行では`benchmark-complete.json`自体がDriveへ保存されていなかった。
再開は元の`plan.json`に残るbenchmarkのdigestと選択済みbatchを保持し、
要約を再生成したことにしない。新しい実行では要約もcampaignへ保存する。
学習・データ・モデルのsourceとTorch版は全保存stageと照合し、各stageの
厳密なresume recipe検証も維持する。復旧controllerの変更は許容し、
元の計画を変更せず`resume-*.json`に今回のコードとcontrollerを記録する。

完了済みのDPTを再学習せず、query T32はupdate 3,001から再開する。
optimizer、RNG、抽出位置を復元し、T32終了後に全重みと対応するoptimizer状態を
query T64へ継承する。T64の冒頭600更新ではCNNを凍結する。
T128は引き続き対象外である。
