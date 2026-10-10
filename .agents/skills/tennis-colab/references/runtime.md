# 状態と復旧の契約

ローカルstateは `~/.local/state/tennis-lab/colab/<session>/`。
`state.json` はskillのmetadata、`cli-sessions.json` は公式CLI所有のregistry、
`identity` はこのsessionのSSH鍵、`jobs/<job-id>/transport.log` は接続・workerのログ。
registry・鍵・rclone資格情報を成果物や会話へ出さない。registryを手編集しない。

VM側の管理領域は `/content/.tennis-colab/<session>/`、通常のcheckoutは `/content/tennis-lab`。
接続先・保存先・Gitの版はstartのJSONで確認する。任意コードはこのcheckout等の明示cwdで実行する。

## 接続と作成を区別する

公式CLI 0.7.4のSSH proxyは、local registryにないsession名を自動作成する。
skillはowned stateとregistryの存在を先に確認し、欠落時はproxyを呼ばない。
startも既存のstate名を上書きしない。読み取りや再接続をGPU新規割当として扱わない。

公式CLIのglobal `--config` / `--auth oauth2` をOpenSSH ProxyCommandにも明示する。
独立したSSH tunnelの同時接続は429になり得るため、ControlMasterと同じControlPathを使う。
コマンド・status・file操作は同じ接続を共有する。セッション終了は明示的なstopが担当する。

## 非同期コマンド

execはVMにrequestを準備し、ローカルからSSH workerをbackgroundで起動する。
workerはそのSSH channelでコマンドの終了を待つ。CLIの提出応答と実際のコマンド完了は別である。
GitのHEAD・差分・選ばれたuntracked sourceも実行記録に残す。コードの正本はローカルGitで維持する。

状態はpending → starting → running → finalizing → completed/failed/cancelled。
Drive保存が失敗した場合はsave_failed。workerが完了receiptを残さず消えた場合はunknownとして扱う。
local SSH dispatcherだけが終了したpendingはdispatch_failedと表示する。

workerは実行前にDriveへ書けることを確認し、実行中はログと指定outputを周期保存する。
コマンドが失敗しても生成済みの成果物とログを残す。最終成果物の保存後に完了statusを公開する。
PIDだけでなくprocess start tokenを保持し、別processへ再利用されたPIDを稼働中と誤認しない。

## 障害からの再開

- **提出・通信が失敗:** 同じjobを再投入せず、local transport.log、remote status、Drive運用記録を調べる。
- **実行が失敗:** returncodeとログを読み、同じVMで必要な調査・修正を行う。修正差分を回収する。
- **保存が失敗:** 原因を直して、停止済みjobをsaveする。保存できていない状態でstopしない。
- **VMが消失:** Drive上の記録・checkpointの保存完了を確認し、新sessionへ同じGit・データ・設定を用意する。
  任意processのメモリ状態は復元しない。学習側が対応するcheckpoint再開を使う。

コマンドにcredentialを埋め込むと実行記録に残るため、資格情報は管理された設定fileや環境の仕組みで渡す。
任意shellの副作用まで一般的に巻き戻すことはできない。再実行前に既存出力と実行状態を確認する。
