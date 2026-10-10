# 操作例

repo rootで実行する。session名・commit・task・run IDは実際の対象へ置き換える。
`--state-dir` と `--colab-bin` はsubcommandより前。既定stdoutはJSON。

```bash
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py doctor --remote
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py start --session court-work --ref HEAD --dry-run
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py start --session court-work --ref EXACT_COMMIT

.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py exec --session court-work --job-id setup -- bash -lc 'uv sync --locked && git submodule update --init third_party/dinov3'
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py status --session court-work --job-id setup
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py logs --session court-work --job-id setup --lines 80
```

GitHubのSSH originは同じrepositoryのcredential-free HTTPS URLへ変換する。秘密をURLへ埋め込まない。
private repositoryのGit認証は自動注入しない。fetchできない場合は理由を確認する。

`start` はGPUを確保してGit checkout・rclone/uvの準備を行う。taskのPython環境や入力データは後続execで準備する。
同名のlocal stateがあればnewで上書きしない。setupの一部が失敗した場合は、状態を調べて `setup --session ...` で残りを再実行できる。
このsetupは、既存checkoutの未commit変更や異なるcommitを上書きしない。

## 学習の入力と保存先

Drive上のdata/ckptを、VMのrepository相対pathへ取得・検証する。
VM内のrclone設定pathは `RCLONE_CONFIG` として管理コマンドへ渡される。
資格情報の中身を表示せず、その環境変数を利用する。

例えば学習出力を `outputs/court_detection/train/example/s42` に置く場合:

```bash
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py exec \
  --session court-work --job-id court-train \
  --runner-output outputs/court_detection/train/example/s42 -- \
  .venv/bin/python -m src.tasks.court_detection.scripts.train \
  'data.source.scene_ids=[B00,B01,B02,B03]' \
  run.output_dir=court_detection/train/example/s42 \
  run.artifact_store.mode=rclone \
  run.artifact_store.remote=gdrive \
  run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/example/s42 \
  run.artifact_store.sync_interval_seconds=60
```

ここで使うremote/rootはsession作成時のDrive設定に合わせる。出力pathはtaskの設定を解決して確認する。
学習runnerのArtifactStoreがcheckpoint保存直後の永続化を担当する。workerはrunner-outputへ並行して書かず、process終了後に残りを保存する。
独自スクリプト等でrunnerを使わない場合は `--persist` の周期保存を使い、完成したcheckpointを一時名からrenameして公開する。
学習再開には対応するcheckpoint・設定・出力先を明示し、別の条件へ変える場合は新runとする。

## ファイル・修正・停止

```bash
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py upload --session court-work ./fix.py /content/tennis-lab/fix.py
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py exec --session court-work -- python3 fix.py
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py diff --session court-work --output /tmp/court-diff-unique
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py cancel --session court-work --job-id court-train
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py save --session court-work --job-id court-train
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py stop --session court-work
```

upload/downloadの上限は16 MiBで、内容hashを照合する。large fileはDrive経由。
diffは `changes.patch`、`untracked/`、元commitと除外pathを記録したreceiptを生成する。
必要なファイルが除外された場合は内容と扱いを確認する。credentialを成果物へ混ぜない。

`cancel` は対象のprocess groupへ終了要求を出す。statusで終了・保存まで確認する。
保存に失敗した場合は原因を直し、停止済みjobに `save` を実行して保存を再確認する。
stopは稼働中jobや保存receiptのないjobがあれば拒否する。
