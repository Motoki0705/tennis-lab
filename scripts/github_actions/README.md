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
- queue学習とCUDAテストを `/var/lib/tennis-lab-actions/gpu.lock` で直列化する。

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
- **Queue local GPU training**: `name`、正確な学習`command`、任意のIssue番号を
  入力する。task固有のコマンドは各task READMEを正とし、isolated checkout内で
  依存関係を再現できる `uv run --locked ...` 形式を使う。

学習Actionのsuccessは「queueへの登録成功」を表す。学習本体は6時間を超えても
systemd queue内で継続し、状態・ログ・run checkoutは次へ保存される。

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

## セキュリティ境界

GitHubは、公開リポジトリのself-hosted runnerではforkのPRからホストを侵害されうると
警告している。このためGPU workflowsには`push`、`pull_request`、
`pull_request_target` triggerを置かず、ownerチェック、repository variable、
Environment承認をすべて必須にしている。学習commandへsecretを含めてはならない。

- [GitHub: Adding self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
- [GitHub: Secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)
- [Microsoft: Use systemd to manage Linux services with WSL](https://learn.microsoft.com/en-us/windows/wsl/systemd)
