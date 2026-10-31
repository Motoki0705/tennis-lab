# シーン5ペアのColab逐次実行・再開手順

#1581 / #1584の実行bundle。担当親がL4の1セッションで、入力照合 → P02の3更新probe →
scene-1〜4の再評価 → P02 / P03 / P12 / P13 / P23それぞれの学習・評価を順に実行する。
管理入口は `campaign.py run`。このPRではCPU検証までを行い、Colabのprobe・本実行は未実施。

session管理と終了判定は[tennis-colab](../../../.agents/skills/tennis-colab/SKILL.md)、
Driveの接続・管理方針は[tennis-drive](../../../.agents/skills/tennis-drive/SKILL.md)を使う。
先行の評価入口・manifestの契約はそれぞれ
[評価README](../../../src/tasks/court_detection/scaling/README.md#shared-synthetic-evaluation-manifest)、
[manifest bundle](../run-court-scene-pair-manifests-v1/README.md)を正本とする。

## 状態を確かめる

初回は保存済み状態がない。再開時は、担当親が前のjobの終了・異常終了・VM消失を確認してから進む。
`exec`の提出receiptだけでは完了しない。VMが残る場合は同じsessionの状態とログを確認し、
`unknown`ならskillの`recover`手順に従う。動いている同じcampaignを重ねて提出しない。

```bash
export RCLONE_CONFIG="$HOME/.config/tennis-lab/rclone-drive.conf"
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py status \
  --session "$PAIR_PREVIOUS_SESSION" --job-id "$PAIR_PREVIOUS_JOB"
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py logs \
  --session "$PAIR_PREVIOUS_SESSION" --job-id "$PAIR_PREVIOUS_JOB" --lines 80
```

`PAIR_PREVIOUS_SESSION`と`PAIR_PREVIOUS_JOB`は前回の提出receiptから設定する。
VM消失時もDriveの小さな状態だけをローカルへ取得できる。初回はこの取得を省く。

```bash
PAIR_REVIEW_DIR=$(mktemp -d)
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download \
  outputs/court_detection/campaign/scene-pair-5500-s384-colab \
  "$PAIR_REVIEW_DIR/state" --verify
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  knowledge/runs/run-court-scene-pair-colab-s42/campaign.py status \
  --state-dir "$PAIR_REVIEW_DIR/state"
```

`progress.json`の`stage / condition / latest_saved_update / started_at / updated_at / status`と、
`done/*.json`の一覧を見る。`status: completed, stage: complete`が全段階の完了。
さらに管理jobの`returncode: 0`と`drive_saved_at`を確認する。
`status: failed`、非0、VM消失は成功扱いしない。記録が古い場合も次のrunがDrive上の世代を再照合する。

## 新しいVMから同じ管理コマンドを再実行する

以下はローカルの、このbundleを含むcommitをcheckoutした専用worktreeから実行する。
初回も再開も同じコマンドでよい。再開には前回と同じソース内容を使う。
`PAIR_REF`を最初の提出時に記録し、親の指示なしで別の実装へ更新しない。
認証済みrclone設定の内容は表示しない。

```bash
export RCLONE_CONFIG="$HOME/.config/tennis-lab/rclone-drive.conf"
PAIR_REF=$(git rev-parse HEAD)
PAIR_SESSION="court-scene-pair-$(date -u +%Y%m%d-%H%M%S)"
PAIR_JOB=campaign
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py doctor --remote
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py start \
  --session "$PAIR_SESSION" --ref "$PAIR_REF" --gpu L4 --dry-run
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py start \
  --session "$PAIR_SESSION" --ref "$PAIR_REF" --gpu L4
```

L4の確保を確認し、次の**1つのexec**を提出する。`bootstrap.sh`が`uv sync --locked --python 3.11`、
`third_party/dinov3`の初期化、Drive入力の取得・照合、残りの全段階を前景で実行する。
`PYTHONDONTWRITEBYTECODE=1`と`TORCHINDUCTOR_COMPILE_THREADS=1`はscript内でも明示する。
同じVMで再提出する場合は、終了と保存を確認してから一意な`PAIR_JOB`へ変えて同じexecを使う。

```bash
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py exec \
  --session "$PAIR_SESSION" --job-id "$PAIR_JOB" \
  --runner-output outputs/court_detection/train/scene-pair-5500-s384-colab-probe/P02-s42 \
  --runner-output outputs/court_detection/train/scene-pair-5500-s384-colab/P02-s42 \
  --runner-output outputs/court_detection/train/scene-pair-5500-s384-colab/P03-s42 \
  --runner-output outputs/court_detection/train/scene-pair-5500-s384-colab/P12-s42 \
  --runner-output outputs/court_detection/train/scene-pair-5500-s384-colab/P13-s42 \
  --runner-output outputs/court_detection/train/scene-pair-5500-s384-colab/P23-s42 \
  --persist outputs/court_detection/evaluate/scene-pair-5500-s384-colab \
  --persist outputs/court_detection/campaign/scene-pair-5500-s384-colab \
  -- bash knowledge/runs/run-court-scene-pair-colab-s42/bootstrap.sh
```

入力は各VMの`/content/tennis-lab`以下に1回だけ取得し、同じVMでの再実行は既存bytesを照合する。
途中の入力転送は`.fetching`へ置き、内容検証後に正式名へrenameする。
datasetの再取得を条件ごとに行わない。元の実画像val registryは
`outputs/court_detection/campaign/scene-diversity-6000-v1/evaluation/registry.json`。
pool・実画像store・重みはDriveの対応する`data/`・`ckpt/`から取得する。

最初の段階で実poolの`load_dataset_version(..., verify_images=False)`を必ず通す。
この引数でも画像shardとそのサイズ・内容検証が必要。ラップトップで既知の幾何の末尾桁差を
許容誤差・metadataだけの検証へ置き換えない。Colabでも失敗したら非0で止まり、
担当親が#1591の担当へ事実と環境を返す。入力取得・照合失敗後にprobeへ進まない。

## 設定・checkpointと完了判定

設定の基準は[完了したscene-2の解決済み設定](../run-court-scene-diversity-colab-scene-2-s42/resolved-config.yaml)。
5条件では`data.source`と`mixed.sources.synthetic_court`のtrain/val manifest、
`run.output_dir`、`run.artifact_store.remote_root`、`run.resume`だけを変える。
artifact storeの他の値も含め、項目ごとの照合でそれ以外の差を拒否する。
初回のresumeはnullで、各ペアは同じ事前学習重みから独立に開始する。
probe専用には3更新・validation/保存間隔3・warmup 0を明示して適用する。
本学習の33,160更新・間隔3,316・warmup 4,974には混ぜない。

`campaign.py configs`は入力取得やGPUなしで5設定の照合結果を表示する。
ローカル証拠は`local-config-audit.json`。実VMの設定とデータ来歴は
`configs/`・`config-audit.json`・`inputs.json`へ保存する。

既存4条件は、前回bundleの`training-completion.json`が指定する最終checkpointだけを取得し、
保存済み設定・来歴・内容・実際の33,160更新を照合する。scene-2は`logs/version_1`、他は`version_0`。
既存の学習・評価treeには書き込まない。新しい9評価はすべて新experimentの出力へ保存する。

途中の新学習は、同条件のDrive inventoryから数値の更新数が最大の
`logs/version_N/checkpoints/update-XXXXXXXX.ckpt`を選んで取得する。
`last.ckpt`・best-val・`.uploading-*`は再開候補にしない。同更新の複数世代、別条件、
未完成/破損ファイル、実更新数・設定・データ来歴の不一致は拒否する。
最新ファイルが壊れていても古い世代へ自動で戻らない。
保存直後の`saved/*.json`がある世代は、その記録の内容とも一致させる。
`run.resume={role:artifact,path:court_detection/train/.../update-XXXXXXXX.ckpt}`で
optimizer・schedulerなどを復元する。最終checkpoint保存済みなら再学習せず、照合して評価へ進む。

`restarts/*.json`に再開checkpoint・更新数・未保存末尾の破棄され得る範囲を記録する。
VM消失後は、保存済み更新より後で次の保存境界までのうち、どこまで実行したかは確定できないため、
`actual_last_update: null`を保持する。最初の開始も更新0からのattemptとして記録する。
mid-epochのDataLoaderの位置・worker RNGを厳密には継続しないという前回と同じ制約がある。
同じ条件と学習予算を保つが、連続実行とのbatch列・数値の完全一致を保証しない。

学習runnerがcheckpointの保存直後にrcloneへ公開し、bundleのhookが保存済みbytesと内容を照合して
`saved/*.json`と進捗を更新する。完了印は成果物の検証・Drive保存の後に公開する。
再実行では完了印の入力identityとcheckpoint/evaluationの内容を照合して飛ばす。
各VMの入力照合は、入力の完了印があっても再度行う。
完了印だけが欠けた最終checkpoint・有効なevaluationは照合・保存して採用する。
未完了の評価directoryは`partial-evaluations/`に保存し、新しいdirectoryで評価し直す。
照合・学習・評価・保存の例外は最上位まで伝播し、後続段階へ進まない。

## 宣言する保存先と出力形式

以下はrepo相対pathで、Driveでは先頭に`gdrive:tennis_lab/`を付ける。

| 用途 | 保存先 |
|---|---|
| 新5学習 | `outputs/court_detection/train/scene-pair-5500-s384-colab/<P02/P03/P12/P13/P23>-s42` |
| probe | `outputs/court_detection/train/scene-pair-5500-s384-colab-probe/P02-s42` |
| 全9評価 | `outputs/court_detection/evaluate/scene-pair-5500-s384-colab/<scene-1/scene-2/scene-3/scene-4/P02/P03/P12/P13/P23>-s42` |
| 設定・完了印・進捗・再開記録 | `outputs/court_detection/campaign/scene-pair-5500-s384-colab` |

学習のtreeは`--runner-output`で宣言し、runnerのrclone artifact storeがcheckpoint保存直後に永続化する。
評価・状態のtreeは`--persist`で宣言する。bundle自身も段階境界で検証付き保存を行う。
Colab workerは`operations/colab/<session>/<job-id>`にもコマンド・ログ・終了/保存receiptを残す。

各評価の`evaluation.json`は`court_data_scaling_evaluation_v2`。
実画像val 2,211枚と合成val 2,439枚を、9条件共通のID・順序・batch 8で検証する。
`--synthetic-eval-manifest`で`eval-4scene.val.subset.json`を指定し、
`splits.synthetic_court-val.scenes`と`scene_macro_mean`、実画像の5指標を保存する。
集計値・順序・最終checkpointの来歴と、評価JSON・定性PNGの内容を完了印に束ねる。

全段階終了後、担当親がjobの`returncode=0`と`drive_saved_at`を確認し、保存済み状態の
`stage=complete / status=completed`を確認してからsessionを停止する。

```bash
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py status \
  --session "$PAIR_SESSION" --job-id "$PAIR_JOB"
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py stop --session "$PAIR_SESSION"
```

## CPU検証

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest -n 2 \
  knowledge/runs/run-court-scene-pair-colab-s42/test_campaign.py \
  tests/unit/tasks/base/training/test_artifact_store.py::test_checkpoint_io_publishes_after_local_serialization \
  tests/e2e/tasks/court_detection/test_evaluate_ablation.py \
  --durations=0 --durations-min=0
```

実行コストとnodeは`test-results.json`を参照する。モデル学習とDriveは小さいファイルfixtureと
偽関数で代替し、checkpoint adapterだけは小さい実torchファイルを使う。
実poolの全経路読込、実9モデルの評価、L4のprobe・学習は担当親の実行工程で確認する。
