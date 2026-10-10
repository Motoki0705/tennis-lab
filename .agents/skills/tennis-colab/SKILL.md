---
name: tennis-colab
description: Run and debug tennis-lab workloads on Google Colab GPUs from the local agent. Use for L4 session management, Git-based remote setup, arbitrary Python or shell execution, long training, logs, remote fixes, and recovery with mandatory Drive persistence.
---

# Tennis Lab Colab

ローカルで開発し、Gitの正確なcommitをColabへ反映して実行する。既定GPUはL4。
Colab上の調査・修正も可能で、変更はローカルへ回収する。
Driveの配置・版・保持・整理は [tennis-drive](../tennis-drive/SKILL.md) とそのreferencesが正本。
このスキルへ管理方針を複製しない。

入口は `.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py`。
実行用job catalogは不要。`exec ... -- <argv>` で任意のPython・シェルを起動する。
シェル構文が必要なら `-- bash -lc '...'` と明示する。

## 準備と接続

1. `doctor --remote` でCLI版、認証、利用枠を確認する。CLIは専用環境の0.7.4を使用する。
   未導入なら `bash .agents/skills/tennis-colab/scripts/install_cli.sh`。
   現在のLinux/WSL CLIを対象とする。Google認証の同意が必要なら利用者へ案内し、tokenを会話へ貼らせない。
2. 専用worktreeでコードを検証・commitし、ColabからfetchできるGitHubの版を用意する。
   `start --dry-run` でrepo/commit/GPU/Driveを確認する。作業ツリーの未commit変更は送られない。
3. 一意なsession名で `start --session ... --ref <commit>`。GPU確保はこの明示操作だけで行う。
   観測GPUが要求と違えば停止して状況を確認し、CPUや別GPUへ切り替えない。
4. `exec` で `uv sync --locked` と必要なsubmodule等を準備する。task READMEに従い、必要な入力をDriveから取得・検証する。

コマンド例は [references/commands.md](references/commands.md)、通信・状態・復旧の契約は
[references/runtime.md](references/runtime.md) を必要時に読む。

## 実行と永続化

- `exec` の応答は提出receipt。成功終了とは扱わず、`status --job-id ...` の終了状態・returncode・`drive_saved_at` を確認する。
- sessionでは通常1つの管理対象コマンドを実行する。状態・ログ・ファイル操作はSSH接続を共用して利用できる。
- 学習前に成果物の出力directoryを決める。repoの共通training runnerでは `--runner-output <repo-relative-output-dir>` と
  `run.artifact_store` のrclone設定を使い、checkpoint保存直後の永続化を有効にする。
  runner以外の任意処理は `--persist <repo-relative-output-dir>` でworkerの周期保存を使う。
  正確な設定例はcommands referenceを参照する。
- Workerはコマンド・ソース差分・ログ・状態をDriveへ保存し、宣言したoutput treeも周期的にコピーする。
  任意プログラムの書き込み途中のファイルまで整合したcheckpointとは見なさない。checkpointは完了後に公開する実装を使う。
- 保存失敗はコマンドを停止して報告する。VM内だけの成果を成功完了にしない。
- 長時間処理を繰り返し起動せず、既存job IDでstatus/logsを確認する。定期監視を依頼された場合は既存のscheduled-followupを使う。
  exec自体が非同期なので、コマンド側を `&` 等でbackground化せず、代表processが処理完了まで待つ形で実行する。

## 調査・修正・終了

失敗時はsessionを保持し、logsと保存先を調べ、同じ環境で修正・再実行する。
任意コードは `exec`、小さなファイルはupload/downloadを使用する。大きなdataset/weightはDriveスキルで転送する。
`diff --output ...` はtracked差分と安全に回収可能なuntrackedファイルを保存する。
ローカルの対象worktreeとcommitを照合し、差分を確認して適用・検証・commitする。既存のローカル変更を上書きしない。

終了前に全jobの終了・Drive保存を確認して `stop`。stopはコード差分もローカルへ退避する。
モデルや学習結果の全量ローカル回収は行わず、依頼されたものだけDriveから取得する。
通信断やVM消失をコマンド成功と扱わない。保存済みcheckpointと対応設定から新しいsessionで再開する。
