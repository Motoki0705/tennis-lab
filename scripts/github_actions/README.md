# Local GPU GitHub Actions runner

この構成は、公開リポジトリ `Motoki0705/tennis-lab` のGPUジョブだけを
WSL2上のself-hosted runnerへ送る。通常のPR CIは引き続きGitHub-hosted runnerで
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
- main、`.gate`、`.slot-0`、`.slot-1`の4つのGPU lockだけを
  `tennis-actions`と`kamimura`の双方から書き込み可能にする。
  親state directoryは`0710`で`kamimura`には通過権限だけを与え、assets、runs、
  training queueの内容は引き続き`tennis-actions`専用とする。

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

- **Local GPU tests**: DINOを初期化してrepositoryのCUDA extensionsをbuildし、
  CUDA preflightの後、`spin test --all --serial -m cuda` を実行する。学習中なら
  GPU lockを取得できず、明示的に失敗する。
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
`state=terminating`のままGPU lockを保持する。MCP container jobはさらにdeterministic
containerの停止ackを必要とする。in-groupのuninterruptible processで不在を証明できない
場合にserviceとcapacityが安全側で停止中のまま残ることは意図した挙動である。
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

## WSL MCP専用trusted runner

MCPのデプロイは通常の`tennis-actions` GPU runnerでは実行しない。MCPのstate、control
directory、canonical checkout、`kamimura`のuser systemdを更新する必要があるため、
`kamimura`として動く専用runnerを使用する。

`gh`がrepository adminとして認証済みのcanonical checkoutで、次を1回実行する。
登録tokenはpipeだけを通り、ファイルやshell履歴には保存されない。

```bash
gh api --method POST \
  repos/Motoki0705/tennis-lab/actions/runners/registration-token \
  --jq .token \
  | scripts/github_actions/install_trusted_mcp_deploy_runner.sh \
      --registration-token-stdin
```

installerは`trusted-mcp-deploy`ラベルを持つrepository runnerと、次のuser serviceを
作成して起動する。

```text
tennis-lab-trusted-mcp-deploy-runner.service
```

runnerのpre-job hookは、`Motoki0705/tennis-lab`の
`.github/workflows/deploy-wsl-mcp.yml`、`main`、repository owner、`deploy` job、
`push`または`workflow_dispatch`がすべて一致する場合だけjobを許可する。不一致のjobは
checkout前に失敗する。通常のCUDAテストと学習は引き続き隔離された
`tennis-actions` runnerで実行する。

既存環境で共有GPU lockの権限だけを再設定する場合は、次を実行する。

```bash
sudo bash scripts/github_actions/install_self_hosted_runner.sh \
  --configure-gpu-lock-only
scripts/github_actions/install_trusted_mcp_deploy_runner.sh
```

## セキュリティ境界

GitHubは、公開リポジトリのself-hosted runnerではforkのPRからホストを侵害されうると
警告している。このためGPU workflowsには`push`、`pull_request`、
`pull_request_target` triggerを置かず、ownerチェック、repository variable、
Environment承認をすべて必須にしている。学習commandへsecretを含めてはならない。

MCP deploy runnerはownerのhomeへ限定的な書き込み権限を持つため、固有labelだけを
security boundaryとして扱わない。workflow側のowner/main/path条件に加えて、host側の
pre-job hookでworkflow pathとGitHub contextを再検証する。hookを無効化した状態で
trusted runnerを起動してはならない。

- [GitHub: Adding self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
- [GitHub: Secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)
- [Microsoft: Use systemd to manage Linux services with WSL](https://learn.microsoft.com/en-us/windows/wsl/systemd)
