# Colab workflow interface

Meiji 3カメラのボール検出には [ConvNeXtUNet / L4 学習レシピ](train/ball_meiji_3cam_l4.md) を使用する。

このディレクトリは、`tennis-lab` の非対話処理をローカル端末からGoogle
Colabへ送る統一入口です。session作成、Drive接続、入力のVM local diskへのstage、
repository環境の構築、処理実行、成果物の検証・Driveへのatomic publish、必要なら
local download、session停止までを `scripts/colab/run.sh` が管理します。学習jobはVM
localへ出力し、runnerがcheckpoint・TensorBoard・設定をrcloneでDriveへ同期します。
入力はVM local diskへcopyします。GUIやOpenCVのinteractive modeはcatalogへ登録しません。
配置の正本と拡張時の監修手順は [STORAGE_LAYOUT.md](STORAGE_LAYOUT.md) を参照してください。

## 前提条件

公式PyPI版 `google-colab-cli==0.6.0` はPython 3.12以上を必要とします。repositoryの
環境とは分離し、`uv` toolとして固定versionをinstallします（公式対応hostはLinuxと
macOSです）。

```bash
uv python install 3.12
uv tool install --python 3.12 \
  --with 'jupyter-kernel-client==0.15.0' 'google-colab-cli==0.6.0'
colab version
```

`google-colab-cli==0.6.0` が使う `jupyter_kernel_client.KernelClient` は依存先の
1.xでは公開されていないため、Python 3.12で検証した `0.15.0` を併せて固定します。
この固定がないと `colab version` やhelpが成功してもruntime接続時に失敗します。
すでに依存を固定せずinstallしている場合は、tool専用環境を次のコマンドで修復します。

```bash
uv tool install --reinstall --python 3.12 \
  --with 'jupyter-kernel-client==0.15.0' 'google-colab-cli==0.6.0'
```

`colab version` が `Version: 0.6.0` 以上を返す必要があります。一方、このworkflowの
local driverはrepositoryの `.venv/bin/python` で動きます。通常の開発環境を作成した
うえで実行してください。worktreeから元repositoryのvenvを使う場合だけ、実体への
absolute pathを明示できます。

```bash
TENNIS_LAB_COLAB_PYTHON=/path/to/tennis-lab/.venv/bin/python \
  bash scripts/colab/run.sh jobs
```

Colab accountの利用枠、選択したGPU、licensed checkpoint（SMPL-X/SMPL-Hを含む）は
利用者が用意します。jobの `setup` hookは `uv sync --locked`、必要なsubmodule、CUDA
extension、NHTをVM内に構築します。

## Drive認証方式

入力とpublish済み成果物は、どちらの方式でも既定ではDriveの
`tennis_lab/` 配下に置きます。

- `--drive-mode mount`は `colab new` の認証後に `colab drivemount` を実行します。
  Drive mount固有のGoogle OAuth URLが端末に表示されるため、browserで同意し、端末の指示に従ってEnterを押して続行します。
  初回のColab CLI認証で求められるcode入力とは別の手順です。人間が操作できる端末向けで、完全headlessでは
  ありません。CLIの対話待ちは最大600秒です。CLIの終了コードだけには依存せず、VM側で
  `/content/drive`が実際のmountpointであり`MyDrive`を提供することを確認してから入力stageや
  学習を開始します。未mountの同名directoryへ出力することはありません。
- `--drive-mode rclone`（既定）はbrowser操作済みのrclone configを使うheadless方式
  で、学習jobを含めてDrive mountの追加承認を必要としません。local hostにも `rclone` が必要です。別端末で `rclone config` を完了して
  configを安全に転送するか、既存configを指定し、owner以外が読めないようにします。

```bash
chmod 600 ~/.config/rclone/rclone.conf
rclone lsd gdrive:
```

`--rclone-config` を省略すると `RCLONE_CONFIG`、次に
`~/.config/rclone/rclone.conf` を使います。指定したremote名の既定は `gdrive` です。
configは0600を検証してから必要なrunの間だけVMの `.secrets/` へuploadし、成功・失敗
にかかわらず削除します。token、OAuth code、secret URL、config本体をrepository、
Hydra override、log、Drive成果物へ書かないでください。

## 実行とlifecycle

catalogとそのaccelerator分類は接続なしで確認できます。

```bash
bash scripts/colab/run.sh jobs
bash scripts/colab/run.sh jobs --json
```

代表的な一連の操作は次のとおりです。

```bash
# headless学習。既存rclone credentialを一時転送する
bash scripts/colab/run.sh run ball_detection \
  --drive-mode rclone --download-to ./colab-artifacts

# 生成・前処理jobも同じ認証方式を使える
bash scripts/colab/run.sh run court_detection_materialize_targets \
  --drive-mode rclone --rclone-config ~/.config/rclone/rclone.conf \
  --download-to ./colab-artifacts

bash scripts/colab/run.sh status <run-id>
bash scripts/colab/run.sh progress <run-id> --watch
bash scripts/colab/run.sh logs <run-id> --tail 40
bash scripts/colab/run.sh resume <run-id> --keep-on-failure --download-to ./colab-artifacts
bash scripts/colab/run.sh download <run-id> --to ./colab-artifacts
bash scripts/colab/run.sh stop <run-id>
```

`--gpu auto`（既定）はGPU jobへT4、CPU jobへCPU runtimeを割り当てます。CPU jobは
`--gpu L4` などでGPU runtimeへ載せられますが、GPU jobを `--gpu cpu` へ落とすことは
できません。`--high-mem` はacceleratorとは独立した要求で、installed Colab CLIの
`new --help` に同capabilityがない場合はsession作成前に失敗します。公式PyPI
`google-colab-cli==0.6.0` の `colab new` はまだ `--high-mem` を提供しないため、この
versionではworkflow側の同flagを付けずに実行します。将来のinstalled CLIが
`colab new --help` に `--high-mem` を公開した場合だけ利用でき、実際にhigh-memory
runtimeが得られるかはColabのplan・quota・在庫に依存します。

`run` は成功時、または `--keep-on-failure` のない既知の失敗時にsessionを停止します。
失敗VMを調査・再実行する場合は初回 `run` に `--keep-on-failure` を付けます。`resume`
は同じrun id・request・保持中sessionだけを再利用し、完了runを変更しません。
再試行中の失敗時もVMを残すには、`resume` にも `--keep-on-failure` を付けます。
入力requestやsource snapshotの転送途中で失敗した場合も、保持済みの元ファイルを再転送します。

`--dry-run` はjob schema、source、argv、Drive/session操作を解決しますが、state directory
を作らずColabやDriveにも接続しません。

```bash
bash scripts/colab/run.sh run plcs_generate_dataset --dry-run
bash scripts/colab/run.sh run slcs --gpu A100 --dry-run -- \
  training.trainer.max_epochs=5
```

`--config-name`、path root、device/GPU、output path、および各manifestの
`protected_override_keys` は利用者override禁止です。末尾の `--` 以降で変更できるのは、
そのjobが所有しないmodel/training parameterだけです。別Hydra configを固定して実行
したい場合は、そのconfig、入力、出力を固定した専用job TOMLを追加します。

## 学習出力と進捗確認

学習job TOMLは `output_storage = "drive"` を必ず指定します。rclone modeではrunnerが
予約済みの`paths.output_root=outputs/colab`へlocal出力し、workflowから注入された
`run.artifact_store`を通じて次のremote treeへ同期します。利用者によるartifact storeの
Hydra overrideやsymlinkによる迂回は許可しません。

```text
<remote>:<drive-root>/colab-live/<run-id>/training/
  config.yaml
  logs/.../checkpoints/...
```

checkpointはlocal保存直後にatomic uploadし、その他の出力は周期同期します。正常終了後の
検証済みbundleは従来どおり別の`colab-runs/<run-id>/`へ公開します。失敗しても
`colab-live`は削除しません。
Driveへの書き込み失敗はエラーとして扱い、VMローカルへの切り替えはしません。
ただし、書き込み途中のファイルまで完全性を保証する仕組みではありません。

`progress` はColab contents APIから小さなJSONを取得し、学習中のkernelへ追加コードを
投入しません。処理phase、実行attempt、更新時刻、job実行中は約5秒ごとの生存情報、
ログ末尾最大80行を表示します。共通TrainingRunnerの学習ではepoch（0始まり）、
global_step、max_epochs、直近batch lossと取得可能なcallback metricsも記録します。
非有限のmetricsは `non_finite_metrics` に名前を表示します。

`--watch --interval 10` は10秒間隔で再取得します（最小2秒）。Ctrl-Cで監視だけを
終了でき、学習は止めません。`logs --tail 40` は直近40行を表示します。全ログはDriveの
attempt別ファイルで確認できます。`heartbeat_age_seconds` と `stale`（30秒超）、
`training_update_age_seconds` は監視更新と学習更新を区別します。compileや長いbatch、
setup中に更新が遅れることもあるため、古い更新を自動で学習失敗とは判定しません。
`status` は成果物公開まで含むworkflow状態の確認に使用します。

liveコマンドは稼働中のVMを必要とします。VM停止後はDriveのJSON・ログを参照して
ください。この機能を追加する前に起動したrunにはprogress JSONがなく、後付けでは
有効になりません。独自entrypointが共通TrainingRunnerを使わない場合、phaseとログは
取得できますが学習metricsは提供されません。

`resume` は保持中VMで同じrequestを再試行する機能であり、最新checkpointからの自動
学習再開ではありません。以前のDrive出力は残し、ログはattempt別に保存します。
checkpointから学習を続ける場合は、検証したcheckpointを新しいjobの入力として宣言し、
`run.resume` を指定してください。修正コミットを使う場合も新規runになります。

## source、stage、成果物

既定の `--source git --ref HEAD` はcredentialを含まないGitHub HTTPS originとrefを
exact commit SHA/treeへ解決します。tracked working treeがdirtyなら拒否し、untracked
fileもremoteには入りません。未commitのroot-repository変更が必要な場合は
`--source snapshot` を使います。snapshotはtracked/untracked/deleted pathとcontent
digestを記録し、secret候補を除外します。

8 MiBを超えるuploadは8 MiBごとに分割して送信します。VMで結合後にSHA-256を照合し、
一致したファイルだけを最終パスへ移動します。大きなsnapshotをColab Contents APIへ
一括送信した際の接続リセットと、base64変換時のメモリ増加を抑えるためです。

submodule worktreeの内容はsnapshot archiveへ入りません。`setup` に `submodules` を
含むjobはparent repositoryのgitlinkにあるexact commitをcredential-freeなGitHub
HTTPS URLからcloneします。したがってdirty/uncommittedなsubmodule変更は実行対象に
できません。submodule側でcommitし、parentのgitlinkを更新し、そのcommitをremote
からfetch可能にしてからsnapshotを作成します。

job TOMLの各 `inputs` はDrive rootからの相対 `source` とVM repositoryからの相対
`destination` を宣言します。全入力をVM local diskへcopyしてSHA-256 manifestを作り、
通常は同じrepository相対pathを指定し、外部library配置だけを明示的なstage adapterとします。
処理後は全 `outputs` が実在することを確認して `artifacts.tar.gz` にまとめます。最終
bundleは次へ一度だけatomic publishされ、同じrun idを上書きしません。

```text
<remote>/tennis_lab/colab-runs/<run-id>/
  artifacts.tar.gz
  manifest.json
  request.json
  status.json
```

local run stateは既定で
`$XDG_STATE_HOME/tennis-lab/colab-runs/<run-id>/`、未設定時は
`~/.local/state/tennis-lab/colab-runs/<run-id>/` です。published provenanceにはjob
definition digest、resolved argv、source commit/treeまたはsnapshot digest、各stage入力
と出力のpath/size/SHA-256、実際に観測したPython・platform・GPU・CUDA・PyTorch・uv
情報が入ります。downloadはarchive sizeとSHA-256を検証し、同名で内容が違うlocal
fileを上書きしません。

rclone modeの `download` は停止済みVMを必要とせず、published `status.json` と
`manifest.json` をDriveから直接検証して取得します。また `run` / `resume` はVM側
status取得に失敗しても、atomic publish済みならrclone remoteからcompleted statusを
回復できます。mount modeの後日downloadはlive sessionを必要とするため、通常は必ず
`run --download-to` で停止前に取得してください。`stop` 後のmount sessionから
`download` することはできません。

大容量入力は1 inputごとに再帰copyされ、output archive作成時にはoutput総量に加えて
1 GiBの空きが必要です。rcloneの入力copy・publish copy/checkにはそれぞれ1時間の
上限があります。VM local diskに「source + staged inputs + archive」（local出力jobではoutputsも）が収まる
ことを先に確認してください。source snapshot自体にも20 GiBのuncompressed上限が
あります。多数のsmall fileを避け、長い学習はcheckpointを含むjob outputやphaseに
分けてください。job timeoutの最大7日はColab runtimeの寿命を保証しません。

## mutable dataset workflow

`writable = false` の入力と重なるoutputは拒否されます。既存datasetへ追記する
`tennis_scene_generate_dataset` と `slcs_precompute_dino_tokens` だけはdataset directory
全体を `writable = true` inputかつ同じexact outputとして宣言します。初回attemptは
canonical Drive inputをstageし、`resume` もfailed attemptのVM内変更を捨てて同じ
canonical inputから再stageします。さらに初回input manifestと一致しなければ失敗する
ため、resume中にcanonical inputを差し替えることはできません。

run outputはcanonical inputをin-place更新せず、run id付きimmutable artifactとして
publishされます。後続jobへ渡すときはartifactをdownload・展開して内容とmanifestを
確認し、該当pathをDriveのcanonical inputへ明示的にpromoteしてから新しいrun idを
作ります。SLCSの標準的な受け渡しは次の順です。

1. `tennis_scene_generate_dataset` の `data/tennis_scene_dataset` をpromoteする。
2. `slcs_make_splits` の `data/tennis_scene_dataset/splits.json` を同datasetへpromoteする。
3. `slcs_precompute_dino_tokens` の完全な `data/tennis_scene_dataset` をpromoteする。
4. `slcs` はpromote済みの同pathをread-only inputとして学習する。

生成runのartifactを確認せずcanonical inputへ自動上書きするfallbackはありません。

## 標準job catalog

| job | accelerator | 固定entrypoint / 用途 |
| --- | --- | --- |
| `ball_detection` | GPU | `src.tasks.ball_detection.scripts.train` / TrackNet学習 |
| `ball_detection_staged` | GPU | `src.tasks.ball_detection.scripts.train_staged` / default TrackNet-only staged phase学習 |
| `court_detection` | GPU | `src.tasks.court_detection.scripts.train` / synthetic Court KP学習 |
| `court_detection_mixed` | GPU | `src.tasks.court_detection.scripts.train_mixed` / synthetic + real mixed学習 |
| `court_detection_materialize_targets` | CPU | `src.tasks.court_detection.scripts.materialize_targets` / SEG・LINE target生成 |
| `blcs_generate_dataset` | CPU | `src.tasks.blcs.scripts.generate_dataset` / single-object dataset生成 |
| `blcs` | GPU | `src.tasks.blcs.scripts.train` / standard学習 |
| `blcs_tracking` | GPU | `src.tasks.blcs.scripts.train --config-name train_tracking` / tracking-query学習 |
| `plcs_generate_dataset` | CPU | `src.tasks.plcs.scripts.generate_dataset` / single-object dataset生成 |
| `plcs` | GPU | `src.tasks.plcs.scripts.train` / standard学習 |
| `plcs_tracking` | GPU | `src.tasks.plcs.scripts.train --config-name train_tracking` / tracking-query学習 |
| `slcs_make_splits` | CPU | `src.tasks.slcs.scripts.make_splits` / recording単位split生成 |
| `slcs_precompute_dino_tokens` | GPU | `src.tasks.slcs.scripts.precompute_dino_tokens` / DINOv3 token生成 |
| `slcs` | GPU | `src.tasks.slcs.scripts.train` / temporal scene-localization学習 |
| `synthetic_data_generation` | GPU | `src.synthetic_data_generation.scripts.run_scene_pipeline` / B00 canonical scene pipeline |
| `tennis_scene` | GPU | `src.tennis_scene.scripts.run_pipeline` / headless multiview統合推論 |
| `tennis_scene_generate_dataset` | GPU | `src.tennis_scene.scripts.generate_dataset` / pseudo annotation追記 |
| `submodules_demo_gvhmr` | GPU | `src.submodules.scripts.demo_gvhmr` / headless video-to-mesh demo |

PLCS dataset generatorの現行configは `run.device=cpu` なのでCPU jobです。全GPU jobは
実行前にNVIDIA GPUとPyTorch CUDAの両方を観測し、どちらかが利用不可なら処理を開始
しません。`tennis_scene` 系は対話的player-association UIを起動せず、Driveから固定の
`data/tennis_scene/player_association_result.json` を読みます。

各TOMLの `inputs.source` がDrive上の正本です。特にDINOv3 weightは
`data/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`、PLCS generator用SMPL-Hは
`data/smplx/smplh/`、GVHMRのface sourceは
`data/smplh/neutral/model.npz` に置きます。GVHMR/tennis-scene checkpoint一式は
`ckpt/`、motion dataは `data/ACCAD/`、scene pipeline用SMPL-Hは `data/smplh/` です。
`synthetic_data_generation` はこれらに加え
`data/synthetic_data_generation/raw/B00.mp4` と
`ckpt/court_detection/line/court-detection-epoch19.ckpt` を必要とします。dataset系は
job名に対応する `data/{blcs,plcs}/...` または `data/tennis_scene_dataset/` をstage
します。正確な最小単位はcatalog TOMLを正本とし、READMEに別のinput schemaを複製
しません。

生成物を次段へ渡す場合もrun artifactを検証してcanonical Drive pathへpromoteします。
代表的な依存は `blcs_generate_dataset` → `blcs`、`plcs_generate_dataset` → `plcs`、
`court_detection_materialize_targets` → `court_detection_mixed` です。mixed学習はpromote
済みの `data/court_detection/derived_targets/` を必須inputとしてstageします。

## job TOMLを追加する

schema version 1は全top-level fieldと、各inputの `writable` を必須とします。

```toml
schema_version = 1
output_storage = "drive"
name = "my_training_job"
description = "One non-interactive training workflow."
accelerator = "gpu"
timeout_seconds = 3600
setup = ["base"]
protected_override_keys = [
  "paths.data_root",
  "paths.output_root",
  "data.scene_dir",
  "run.output_dir",
  "run.artifact_store",
  "run.gpus",
]
outputs = ["outputs/colab/my_training_job"]

[command]
module = "src.package.scripts.train"
args = [
  "paths.output_root=outputs/colab",
  "run.output_dir=my_training_job",
  "run.gpus=1",
]

[[inputs]]
source = "data/my_dataset"
destination = "data/my_dataset"
writable = false
```

`setup` は `base` から始め、必要時だけ `submodules`、`cuda_ops`、`nht` をこの順で
追加します。commandはPython moduleまたはshell-free argvとし、`bash -c`、secret、
Drive FUSE pathを直書きしません。input/output宣言はsymlinkなしのstrictなrepository-relative
POSIX pathです。実commandが書く全declared outputを固定し、そのpathを決めるHydra
keyを `protected_override_keys` に含めます。writable inputは同じexact pathをoutputに
宣言し、read-only inputとoutputは重ねません。追加後はloaderとdry-runで検証します。

```bash
bash scripts/colab/run.sh jobs
bash scripts/colab/run.sh run my_training_job --dry-run
```

## 既存timestamp固定runner

`train/20260829T150257Z/run_b01_b03_alignment.sh` はgeneric catalogとは別の、
B01→B02→B03 3DGS reconstruction / court alignment検証を再現する固定runnerです。
Drive mount、入力SHA検証、locked依存、NHT、DINOv3、GPU/CPU処理、scene単位atomic保存
を自身で所有するため、generic jobへ重複登録しません。

```bash
bash scripts/colab/train/20260829T150257Z/run_b01_b03_alignment.sh
bash scripts/colab/train/20260829T150257Z/run_b01_b03_alignment.sh --dry-run
```

検証結果、GPU要件、Drive layoutは
[`train/20260829T150257Z/REPORT.md`](train/20260829T150257Z/REPORT.md)を参照して
ください。各taskのmodel/data schemaはtask READMEを正本とし、このREADMEはColab
lifecycleとjob interfaceだけを管理します。
