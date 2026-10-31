# Local GPU GitHub Actions runner

この構成は、公開リポジトリ `Motoki0705/tennis-lab` のGPUジョブと、親issue書き直し用のagyを
WSL2上のself-hosted runnerで実行する。通常のPR CIは引き続きGitHub-hosted runnerで
実行し、ローカルrunnerはリポジトリownerが手動実行したジョブだけを受け付ける。

## GitHubでの1回限りの設定

1. Repository **Settings → Actions → Runners → New self-hosted runner** を開き、
   **Linux / x64** を選択する。表示された `config.sh` コマンド中の
   1時間有効な登録トークンを、次節のインストーラーへ貼り付ける。GitHubが表示する
   download/configコマンド自体は実行しない。
2. **Settings → Environments → New environment** で `local-gpu` を作る。
   Required reviewersへ `Motoki0705` を追加し、Deployment branchesは
   `Selected branches and tags` の `main` のみにする。自分自身で承認できるよう、
   `Prevent self-review` は有効にしない。
3. **Settings → Secrets and variables → Actions → Variables** にrepository variable
   `LOCAL_GPU_ACTIONS_ENABLED=true` を追加する。この変数がない間、GPU jobは
   runnerへ割り当てられずskipされる。
4. **Settings → Actions → General → Workflow permissions** が
   `Read repository contents and packages permissions` であることを確認する。
   2026-08-05時点では、すでにこの設定になっている。

登録トークンはsecretとして保存したり、チャットへ貼り付けたりしない。
GitHubの公式手順でも登録トークンの有効期限は1時間とされている。

## WSLでの1回限りの設定

GitHub画面で登録トークンを表示したまま、リポジトリrootから実行する。

```bash
sudo bash scripts/github_actions/install_self_hosted_runner.sh
```

sudoパスワードの後、別の非表示プロンプトでGitHub登録トークンを求められる。
インストーラーは次を行う。

- sudo/docker権限を持たない `tennis-actions` system userを作る。
- GitHub公式release APIが返すSHA-256 digestを検証してrunnerと`uv`を導入する。
- `data/` と `ckpt/` だけをrunner用namespaceへread-only bind mountする。
- runnerとtraining queueをsystemd serviceとして有効化する。
- runnerから通常ユーザーのhome、検出したWindowsドライブ（`/mnt/c` など）、
  WSLg mountを見えなくする。WSLのDNSが参照する `/mnt/wsl` は遮断しない。
- queue学習とCUDAテストを `/var/lib/tennis-lab-actions/gpu.lock`
  から導出される2-slot protocolで調停する。CUDA CIは従来どおり
  main lockをexclusiveに取得し、`all`として動作する。
- main、`.gate`、`.slot-0`、`.slot-1`の4つのGPU lockを
  `tennis-actions`専用として作成する。

queue runtimeの配置は `install_queue_bundle.sh` に集約している。履歴helperや
共有Python moduleの追加を既存runnerへ反映するには再インストールが必要。
実行中workerの切替と履歴整理の手順は
[Training Queue skill](../../.agents/skills/training-queue/SKILL.md#filenames-and-automatic-history-retention)
を参照し、worker終了後に更新する。

WSLのsystemd serviceだけではWSL instance自体を維持できないため、Windowsログオン時に
WSLへのhandleを保持するTask Scheduler taskも登録する。

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$(wslpath -w scripts/github_actions/register_wsl_keepalive.ps1)"
```

Windowsがsleepまたはshutdownしている間はrunnerもofflineになる。常時受付が必要なら、
Windowsの電源・sleep設定もそれに合わせる。

## 使い方

GitHubの **Actions** タブから次を手動実行する。

- **Local GPU tests**: `target=cuda`（既定）ではDINOを初期化してrepositoryのCUDA extensionsをbuildし、
  CUDA preflightの後、`spin test --all --serial -m cuda` を実行する。学習中なら
  GPU lockを取得できず、明示的に失敗する。rendered alignmentとexported player checkpointの
  opt-inも明示的に有効にする。NHTの公開コマンド・B00の保存済みalignment・playerのcheckpointと
  保存済み予測など、各テストが要求する資産/実行環境の準備が必要。
  `target=local-data` では `test_file=tests/integration/submodules/test_checkpoint_loading.py`
  のように実在する1ファイルを指定する。CPU backend・CUDA不可視・native thread各1で
  `local_data and not cuda` を直列実行し、GPU lockやCUDA buildは使わない。
  両方とも `-p scripts.ci_metrics.require_execution` でskip・収集時skip・0件を失敗にし、
  `.ci-acceptance/` のJUnitと再現出力をartifactへ保存する。owner・variable・environmentの
  制限と信頼されたmain refは両モードで共通。
- **Queue local GPU training**: `name`、正確な学習`command`、任意のIssue番号、
  `resource` (`all`/`half`、既定`all`)を入力する。task固有のコマンドは各task READMEを正とし、isolated checkout内で
  依存関係を再現できる `uv run --locked ...` 形式を使う。

`half`は同じGPU上の論理slot予約であり、MIG分割やVRAM上限ではない。
queueは`CUDA_VISIBLE_DEVICES`を変更しないため、2つの`half` jobが同時にGPU全体を
参照する。各jobが同時実行に収まることを利用者が確認する。

学習Actionのsuccessは「queueへの登録成功」を表す。学習本体は6時間を超えても
systemd queue内で継続し、状態・ログ・run checkoutは次へ保存される。

queue serviceの停止時、systemdはservice cgroupへSIGTERMを送るが、強制timeoutや
SIGKILLではwrapperを打ち切らない。各wrapperはqueueが作成・検証したprivate PGIDを
所有し、15秒のTERM grace後にKILLへescalateして、group不在とleader reapを証明するまで
`state=terminating`のままGPU lockを保持する。in-groupのuninterruptible processで
不在を証明できない場合にserviceとcapacityが安全側で停止中のまま残ることは意図した挙動である。
自己daemon化や`setsid`でqueue PGIDからescapeするhost processはsupport対象外である。

```text
/var/lib/tennis-lab-actions/training-queue/
/var/lib/tennis-lab-actions/runs/<github-run-id>-<attempt>/
```

ローカルでの確認コマンド:

```bash
sudo systemctl status 'actions.runner.*' tennis-lab-training-queue.service
TRAINING_QUEUE_DIR=/var/lib/tennis-lab-actions/training-queue \
  /opt/tennis-lab-actions/bin/training_queue.sh status
```

## agyの導入と疎通

親issueの書き直しは `tennis-actions` のGoogleアカウントログインを使う。
固定版の導入とログインはデスクトップで人間が行う。GPU runnerの導入スクリプトとは独立している。
Linux x64のversion・版付きURL・sha256・モデルの正本は
[`agents.conf`](../../.agents/ops/lib/agents.conf) の `AGY_VERSION`, `AGY_URL`,
`AGY_SHA256`, `ISSUE_REWRITE_AGY_MODEL`。環境変数でこれらを上書きしない。

### 導入とログイン

runnerの導入済みデスクトップで、レビューしたmainのcheckoutから実行する。

```bash
sudo bash scripts/github_actions/install_agy.sh
sudo -u tennis-actions env HOME=/var/lib/tennis-lab-actions/home \
  AGY_CLI_DISABLE_AUTO_UPDATE=true \
  /var/lib/tennis-lab-actions/home/.local/bin/agy
```

導入スクリプトは公式GitHub Releaseの固定assetを取得し、SHA-256が一致してから中の `antigravity` を `agy` として配置する。
所有者は `tennis-actions`、置き先は `/var/lib/tennis-lab-actions/home/.local/bin/agy`。
同じpinと導入記録があり、実行ファイルも改変されていなければ再取得しない。
検証不一致では新規配置せず、既存の実行ファイルも置き換えない。公式のlatest用 `install.sh` は使わない。

2つ目のコマンドで開いた対話画面で、利用するGoogleアカウントへ直接ログインする。
通常ユーザーのログイン情報、OAuth token、API keyをrunnerへコピーしたり、repository secretに保存したりしない。
runner serviceと同じUnix user・HOMEでログインする。認証が切れたときも、この対話コマンドで再ログインする。
認証の画面と保存の仕組みは[公式のCLI認証手順](https://antigravity.google/docs/install?tab=cli)を参照。

### 自動更新と固定版の更新

2026-10-31に[公式のself-updater説明](https://antigravity.google/docs/cli/troubleshooting/#resolve-self-updater-locks-and-failures)で、
`AGY_CLI_DISABLE_AUTO_UPDATE=true` により自動更新を無効化できることを確認した。
取得した固定版の実行ファイルも、実行せずに文字列を調べ、同名の環境変数を含むことを確認した。
上の対話ログイン、疎通workflow、共通部品はすべてこの変数を設定する。
別の端末で手動起動するときも、同じ環境変数を付ける。

共通部品は自動更新の停止だけに頼らず、各headless呼び出しの前後で `--version` を照合する。
版が変わっていたら失敗し、別の版で続行しない。ログイン後に疎通を実行して、この確認も通す。
更新するときは、[公式GitHub Release](https://github.com/google-antigravity/antigravity-cli/releases)の
tagとLinux x64 assetのURL・`digest`（`sha256:`）をrelease APIから取得する。

```bash
gh api repos/google-antigravity/antigravity-cli/releases/tags/1.4.11 \
  --jq '{tag: .tag_name, asset: [.assets[] | select(.name == "agy_cli_linux_x64.tar.gz") | {url: .browser_download_url, digest: .digest}]}'
```

新しい版を採用するときは上のtagを変更し、返されたassetを取得してSHA-256の一致と
tarball内の実行ファイル名を確認してから、同じPRで `agents.conf` を更新する。
merge後に導入コマンドを再実行し、疎通が成功してから書き直しに使う。

### 手動の疎通

ログイン後、repository ownerがGitHubの **Actions → AGY smoke → Run workflow** で **main** を選ぶ。
[`agy-smoke.yml`](../../.github/workflows/agy-smoke.yml) は手動起動だけを受け付け、ownerとmainをjob開始前に確認する。
checkoutのcredentialは永続化せず、GitHub tokenの権限はcontents・issues・pull requestsの読み取りだけにする。
既存の `self-hosted, linux, x64, tennis-lab` runnerを使うため、GPUジョブの実行中やデスクトップの停止中は待機する。

workflowは軽量なjsonschema環境で `agy_smoke.py` から共通部品を呼び、
`.github/ISSUE_TEMPLATE/06_parent.md` の `##` 見出しを読み取り、実際のファイルと結果を照合する。
job summaryに観測した版・モデル、usage、CLIの所要時間と版の確認を含む所要時間、`denied_actions` を出す。
失敗時は失敗の分類と取得できた拒否一覧を残す。log・prompt・会話ID・認証情報をartifactへ公開しない。
認証・モデル・版・構造化出力のいずれかで失敗したら修正して手動でやり直す。
この疎通だけでは書き直しの本体を有効化しない。

続くpermissions probe段階は、使い捨ての作業領域で読み取りの成功と、危険なコマンドの拒否・ファイル不変を確認する。
形式と判定方法・呼び出し数・上限時間は[共通部品の読み取り権限と実機プローブ](../../.agents/ops/lib/README.md#agyの読み取り権限と実機プローブ)を参照する。
両段階が成功してから実機の疎通成功とし、実行は #1599 で人間が行う。

共通部品の関数・例外・結果の型と読み取り権限の契約は
[ops lib README](../../.agents/ops/lib/README.md#agyの共通呼び出し部品)を正本とする。
ローカルのテストは本物のagyを起動せず、実行ファイルの偽物と小さなtarballだけを使う。
導入テストでは `install_agy.sh --bin-dir DIR --config-file FILE` で一時ディレクトリと
`file://` の偽tarballを指定できる。明示した置き先には現在の実行者の所有権で置き、sudoを要しない。

## セキュリティ境界

GitHubは、公開リポジトリのself-hosted runnerではforkのPRからホストを侵害されうると
警告している。このためGPU workflowsには`push`、`pull_request`、
`pull_request_target` triggerを置かず、ownerチェック、repository variable、
Environment承認をすべて必須にしている。学習commandへsecretを含めてはならない。

- [GitHub: Adding self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
- [GitHub: Secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)
- [Microsoft: Use systemd to manage Linux services with WSL](https://learn.microsoft.com/en-us/windows/wsl/systemd)
