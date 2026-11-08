# X3D/CSN＋mHC queryの容量アブレーション（#1675）

Issue: https://github.com/Motoki0705/tennis-lab/issues/1675

子issueは作成しない。seed 42、720p、RGB＋MDD 5ch、固定snapshot/split、128 frames/updateを維持し、
CNN+DPT T32を10,000更新した共通重みから、mHC query T64の7条件を独立に各6,000更新する。
物理予算は計52,000更新。各queryの来歴16,000更新を物理費用として7回加算しない。

## 学習前に確定した構成

- CNNはX3D/CSN型、幅16/32/64/128/256/512、stem以降のblock数1/2/3/3/2、展開率2。
  downsamplingは時間を混ぜない。各残差blockのdepthwise 3×3×3だけが時間を混ぜ、最深特徴の半径は11。
- decoder幅128のDPTで新たに事前学習し、既存の共通checkpointを流用しない。
- 各frameの4本のqueryは独立した学習可能な初期値。同じdimの4本→mHCで1本→実時間RoPE＋SwiGLU→4本への書戻し。
  AttentionとFFNを1つのmHC区間にし、書戻しには区間の残差なし更新を渡す。最終読出しは4本の平均→LayerNorm→Linear。
- mHCはcoefficient_dim=64、sinkhorn_iters=20、eps=1e-6、residual_identity_bias=4、update_scale_init=0。
- dim 256/384/512/768/1024・4層と、dim512・6/8層。head幅64、FFNは8/3 dimの64単位切上げ。
- activation checkpointingとframe chunkは無効。メモリ対策を使う前にユーザーへ相談する。OOM時に自動変更しない。
- #1514の共通事前学習直後のquery段のLR・optimizerレシピを採る。
  共通DPTはwarmup500、cosine horizon28000。queryは共通10k終端のLRから、warmup0、horizon12000・offset0の最初の6000更新。
  CNN凍結600更新、解除後CNN LRはheadの0.1倍。CNNのAdamW状態だけを名前で継承し、新しいTransformer状態は新規。
  6,000更新で終了する条件に合わせてcosine horizonを黙って6,000へ変えない。
- 主指標はmean source px。median・P95・≤20px率を副指標とする。sceneモデルの採用は変更しない。

## データ・環境

最初にDrive mountの`data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1`をローカルSSDへ取得する。
配布manifestの2,925 payload全件のサイズと、独立2系統のSHA-256を照合する。
取得には#1514の`download_inputs.py`を再利用した。今回の並列数は12（元コードは4）で、データや検証条件は変更しない。
実行ログと最終receiptは`outputs/ball_detection/precompute/capacity_scaling/`へ保存する。

Pythonは`.venv/bin/python`。Python 3.11、`uv.lock`の依存、L4を使用する。
入力の準備は#1514の`run.py prepare`で、同じ30,478 observed-containing training packsと固定validationを再生成する。
train/validationの注釈・split・RGBを変更しない。validationは128,668 owner frames、位置誤差の分母は85,473 observed frames。

```bash
.venv/bin/python -B knowledge/runs/run-ball-sequential-length-20261025/download_inputs.py
.venv/bin/python -B knowledge/runs/run-ball-sequential-length-20261025/run.py prepare \
  --workspace-root /content/tennis-lab
```

元のdownloadスクリプトは空き容量に全量分を要求する。取得済みファイルが多い同一VMでの再実行では、
データを削除せず、不足分の容量と既存ファイルの検証を確認してから実行する。
新VMではデータを最初から検証付きで取得する。

## CPU検証・理論MAC

モデル・転送・再開のテスト設計と実測は`validation.json`へ記録する。
`theory.py`は実際のConv/DPTの出力shapeをmeta forwardで数え、query/mHCは設定・weight形状から行列積のMACを計算する。
正規化・活性化・MDD・入出力・backwardは含めず、実測秒数をGMACから代用しない。

```bash
.venv/bin/python -B knowledge/runs/run-ball-capacity-scaling-20261103/theory.py \
  --output knowledge/runs/run-ball-capacity-scaling-20261103/theory.json
```

## GPU実測と本学習

GPUコマンドは元repoの共有training queueを使い、`--resource all`で逐次に実行する。
専用worktreeのカレントディレクトリから登録する。provider/sessionの帰属を必ず付ける。
以下のPythonコマンドはqueueへ登録する本体であり、直接同時起動しない。
Drive mountを指す明示的な`mounted_drive` rclone aliasと設定ファイルを用意する。認証情報はbundleに含めない。

```bash
PATH=/tmp/ball-tools:$PATH .venv/bin/python -B knowledge/runs/run-ball-capacity-scaling-20261103/run.py benchmark \
  --workspace-root /content/tennis-lab --run-id benchmark-20261103-a \
  --rclone-config /tmp/ball-drive-mount.conf

PATH=/tmp/ball-tools:$PATH .venv/bin/python -B knowledge/runs/run-ball-capacity-scaling-20261103/run.py train \
  --workspace-root /content/tennis-lab --run-id s42-20261103-main \
  --rclone-config /tmp/ball-drive-mount.conf \
  --benchmark /content/tennis-lab/outputs/ball_detection/train/capacity_scaling/benchmark-20261103-a/benchmark.json
```

benchmarkは実データ・拡張・128 frames/updateでDPT BS1/2とquery7条件を確認する。
queryでは凍結CNNのforward/backwardも1更新実行し、定常時間はjoint更新の最初の5更新を除いて求める。
benchmarkの重みを本学習へ転用しない。8構成が成功しなければ本学習を開始しない。
理論計算量・peak allocated/reserved VRAM・定常s/updateを並べ、全52k学習更新の時間を推定する。
定常時間は更新内の入力待ち・拡張・forward/backward・optimizerを含む。診断更新は学習予算と別に記録する。
データ検証・compile・validation・Drive保存の時間は推定の対象範囲を明示して別に扱う。

## 起動担当から独立したqueue service

この環境では有限の`agent_exec` supervisorが終了時に子孫を回収するため、Codex起動担当の内側からqueue workerをdaemon化しない。
親側で、空の共有queueを次のforeground serviceから起動しておき、Codex子は既存workerへの登録と最初の周期保存までを担当する。
`queue_service.py`はsubreaperとしてSIGCHLDを受けて孤児を回収し、queueのPGID終了確認がzombieで止まることを防ぐ。
モデル・予算・メモリ設定は変更しない。service自体はGPU処理を起動せず、queue登録があるまで待機する。

```bash
TRAINING_QUEUE_DIR=/content/tennis-lab/.training_queue .venv/bin/python -B \
  knowledge/runs/run-ball-capacity-scaling-20261103/queue_service.py -- \
  bash .agents/skills/training-queue/scripts/training_queue.sh serve --idle-timeout 3600
```

実運用では親がこのserviceを独立processとして起動し、ログとPIDを保持する。
既存workerがいる場合は重複起動しない。起動担当の終了後もserviceは残り、training queueの停止・取消は通常のCLIで扱う。

## 保存・再開

### 新VM向けの追補（2026-11-04、attempt-3以降）

元の`run.py train --resume`は、保存JSONの配列listとdataclassのtupleを直接比較して停止した。
24項目の型表現以外は計画全体・モデルsourceが一致し、学習更新前に失敗した。
`resume_json_plan.py`は元controllerの`asdict` bindingだけをJSON正規化する明示的な復旧ランチャーである。
モデル設定オブジェクト・学習module・stdlib・hash関数は変更せず、元の全guardを維持する。
元controllerの実hashを保存済みplanとbenchmarkの両方へ照合してから適用する。追加の実効コードと由来はJSONで記録する。
CPU10件で実際の再開境界の通過、予算・model・source・controller変更の拒否、bindingの局所性を確認した。

元の[Issue引き継ぎ手順](https://github.com/Motoki0705/tennis-lab/issues/1675#issuecomment-6606918328)の
データ全件検証、Git bundle・専用worktree・環境の復元、`recover.py`による最新checkpoint復元は引き続き使う。
**再開コマンドだけを下記ランチャーへ変更する。元の計画・checkpointを編集してguardを回避しない。**

追加ランチャーは元の不変`inputs-v1/source.bundle`には含まれない。
Driveの同じcampaign rootに、次の不変snapshotと外部receiptを保存した。

- `recovery/attempt-3-inputs/`、`recovery/attempt-3-inputs-receipt.json`: ランチャー・CPU検証・由来の7ファイル。
- `documentation/resume-attempt-3/`、`documentation/resume-attempt-3-receipt.json`: このREADME、追加コード、旧500更新のknowledge、共通10k完了・今回の再開証拠、未commit差分。

新VMでは元bundleのworktreeへ戻した後、少なくとも追加ランチャーを以下で検証付き復元する。
`/tmp`の復元先が既にある場合は、上書きせず既存内容とreceiptを確認する。
コマンドのcwdは元の専用worktree、PATHはrcloneを含むものとする。

```bash
PATH=/tmp/ball-tools:$PATH .venv/bin/python -B - <<'PY'
import json
import shutil
from pathlib import Path
from src.utils.artifact_store import RcloneSnapshotStore

relative = "outputs/ball_detection/train/capacity_scaling/s42-20261103-main/recovery"
drive = Path("/content/drive/MyDrive/tennis_lab")
receipt = json.loads((drive / relative / "attempt-3-inputs-receipt.json").read_text())
assert receipt["schema"] == "verified_artifact_snapshot.v1"
assert receipt["status"] == "verified" and receipt["generation"] == "attempt-3-inputs"
assert receipt["remote"] == f"mounted_drive:{relative}/attempt-3-inputs"
destination = Path("/tmp/ball-capacity-attempt-3-inputs")
RcloneSnapshotStore(destination, remote="mounted_drive", remote_root=relative,
    config_path=Path("/tmp/ball-drive-mount.conf")).restore(
        "attempt-3-inputs", manifest_sha256=receipt["manifest_sha256"])
for source in (destination / "knowledge").rglob("*"):
    if not source.is_file():
        continue
    relative_file = source.relative_to(destination)
    target = Path.cwd() / relative_file
    if target.exists():
        assert target.read_bytes() == source.read_bytes(), f"Existing file differs: {relative_file}"
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
print("Additional resume inputs restored and verified; training not started.")
PY
```

knowledge全体を復元する場合はdocumentation側の外部receiptから同じAPIで別の空ディレクトリへ復元し、
差分を確認して専用worktreeへ反映する。未知の既存変更を上書きしない。Gitのauthor未設定のため、今回の差分は
commit済みとは扱わず、documentation snapshotの`repro/knowledge.patch`と全変更ファイルで保全している。

起動は親が外部queue serviceを保持し、共通ランナーの既定Codex子へ委任する。
共有queue・GPU実プロセスと既存attemptを照合し、次回は新しいattempt key（次はattempt-4）を使う。
以下はqueueへ渡すコマンド本体であり、親が直接GPUを起動しない。

```bash
PATH=/tmp/ball-tools:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore \
.venv/bin/python -B knowledge/runs/run-ball-capacity-scaling-20261103/resume_json_plan.py train \
  --workspace-root /content/tennis-lab --run-id s42-20261103-main \
  --rclone-config /tmp/ball-drive-mount.conf \
  --benchmark /content/tennis-lab/outputs/ball_detection/train/capacity_scaling/s42-20261103-main/benchmark.json \
  --resume
```

今回の再開は1,500→2,000更新の全500更新でloss/勾配がfinite、次の周期checkpointのDrive全7 payloadを
独立復元して照合した。起動担当の実終了後も2,097更新までの継続を親が一度確認した。
[再開と保存の証拠](resume-attempt-3-startup.json)を参照する。これは全8段52k更新の完了ではない。
checkpoint選択は引き続き最新の検証済みreceiptを使い、上記の更新数へ固定しない。
VMと共に消えるqueue/監視/通知先は新VMで登録し直し、起動担当は最初の周期保存までで終了する。
その後は親が終了イベントだけを監視し、定期的な進捗ポーリングはしない。

正本は`outputs/ball_detection/train/capacity_scaling/s42-20261103-main/`。
`plan.json`を学習前に固定し、コードのGit bundle・全prepared manifest・benchmark・理論計算量を
検証付きの不変inputs snapshotへ保存する。500更新ごとに完全checkpoint・weights・設定・manifest・ログを
世代別snapshotへ保存し、receiptを`persistence/<stage>/`へ残す。Drive保存失敗は実行失敗として扱う。

同じVMでは上のtrainコマンドに`--resume`を付け、active jobがないことを確認して新attemptとしてqueueへ登録する。
完了済みstageはcheckpoint hash・予算・設定・source・validationを照合してスキップする。
再開はoptimizer・RNG・sample positionを保存地点へ戻す。Tやdimを書き換えて同じrunの再開とは扱わない。

新VMではデータを先に取得し、保存時と同じコード・Python/Torch・worktreeの絶対pathを復元してから次を実行する。

```bash
PATH=/tmp/ball-tools:$PATH .venv/bin/python -B knowledge/runs/run-ball-capacity-scaling-20261103/recover.py \
  --workspace-root /content/tennis-lab --drive-root /content/drive/MyDrive/tennis_lab \
  --run-id s42-20261103-main --rclone-config /tmp/ball-drive-mount.conf
```

復旧はreceiptのmarker hashからpayload全件を検証する。最新receiptの破損・同更新の曖昧な候補・既存の異なるローカル状態では停止する。
初期化だけが保存され、更新checkpointがないstageも自動再初期化せず確認を要求する。
長時間ジョブの起動はrepoのparent-session手順に従い、Codex workerが最初の500更新のDrive保存まで確認する。

## 集計

全8 stageの完了後、`report.py`が同じ採点集合とquery7条件の抽出・拡張の一致を照合し、JSON・CSV・表・グラフをDriveへ保存する。
`COMPLETED.json`は全学習の完了、`ANALYSIS_COMPLETED.json`は集計とDrive保存の完了を表す。
ディレクトリや途中checkpointの存在を完了扱いしない。knowledgeの結果ノードは実測に基づいて記録する。
容量2倍あたりの改善量は、CNNを含む総パラメータ比のlog2で誤差改善量を割る。GMAC比でも別に計算する。

seed42の各1回であり、条件差がseedのばらつきの範囲かは判断できない。独立testは未評価。
CNNが異なる#1367/#1514との直接の優劣比較は行わない。仮説を支持しない結果も含め、採用判断と実験結果を分ける。
