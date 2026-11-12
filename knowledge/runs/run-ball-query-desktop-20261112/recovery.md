# 保存状態の確認と再開

稼働中のjobへ再投入しない。今回の登録済みjob・実commandは出力rootの
`production-registration.json`、状態は `production-status.json`、
終了イベントは `notifications/` とsystemd serviceにある。
対話workerの終了後もserviceとtrainerはsystemdの子として継続する。

```bash
systemctl --user status tennis-i1899-length-desktop-a1.service
cat "$ROOT/production-registration.json"
cat "$ROOT/production-status.json"
cat "$ROOT/production/length-s2-T64/latest.json"
```

異常終了・host再起動後は、原因と旧writerの不在を確認する。FAILEDからの自動再試行はない。
新しいattempt名・保存先で再開し、旧世代・不明な末尾log・失敗情報を保持する。
`desktop.py prepare` や `service.py wait-and-serve` のa1登録を再実行する方法は使わない。

再現に必要な入力はDrive root相対の既存run
`outputs/ball_detection/train/query_controls/s42-20261111-main/` にある。
`inputs/inputs-v1/` が固定source bundle・lock・plan・checkpoint来歴の正本。
デスクトップの実行adapter、実command、環境freeze、テスト、状態移管、checkpoint参照は
`desktop-attempts/20261112-a1/documentation/production-start-v1/` に検証保存している。
元のデータ集合は `data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1/`。
元storeからの再利用と全件照合はこのrunの `rebuild_dataset.py` / `verify_dataset.py` を使う。

次はCPUだけの復元例。`$PY` は固定lockのenv、cwdは固定sourceのworktree、
`$RCLONE_CONFIG` は認証済み設定の実パス、`$RECEIPT_DIR` は回収したstageのpersistence directory、
`$RESTORE_DIR` は存在しない新しいdirectoryに設定する。
receipt directoryの取得はtennis-drive CLIの `download-tree` で内容照合する。
最高更新でcompleted世代を優先し、最高世代が重複・不正なら停止する。古い世代へ戻さない。

```bash
CUDA_VISIBLE_DEVICES='' "$PY" - "$RECEIPT_DIR" "$RESTORE_DIR" "$RCLONE_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, 'knowledge/runs/run-ball-query-controls-20261111')
from control import select_receipt
from src.utils.artifact_store import RcloneSnapshotStore
from src.utils.checksum import dual_sha256

receipt_path, receipt = select_receipt(Path(sys.argv[1]))
remote, location = receipt['remote'].split(':', 1)
assert remote == 'gdrive'
store = RcloneSnapshotStore(
    Path(sys.argv[2]), remote=remote,
    remote_root=location.rsplit('/', 1)[0], config_path=Path(sys.argv[3]))
result = store.restore(receipt['generation'], manifest_sha256=receipt['manifest_sha256'])
latest = json.loads((Path(sys.argv[2]) / 'latest.json').read_text())
assert dual_sha256(Path(sys.argv[2]) / latest['checkpoint']) == latest['sha256']
print(json.dumps(dict(receipt=str(receipt_path), latest=latest, restore=result), indent=2))
PY
```

復元後は、固定source・torch・model・budget・runtime・augmentation・manifest・log prefixを照合し、
full checkpointの `resume` から再開する。重みだけの再初期化やAdamWリセットは行わない。
完了済み前半は再学習せず、後半の保存があればその最新full stateを選ぶ。
段境界で後半を新規に始める場合だけ、前半6kの全重み・named AdamW・sample位置を継承し、
旧基準と同じseed42構築とoffset6,000／600更新凍結を維持する。

現行a1 adapterはこの初回移管を固定したものなので、異常終了後の新attemptでは
出力pathとremote rootを明示した同一recipeの再開入口を準備・検証する必要がある。
固定runnerの `read_resume` と、このrunのmodel/AdamW/RNG/sample/log照合を通す。
移管でdata rootが変わる場合は、`prepare` と同様にコピー上のpathと派生manifest digestだけを
変更して全残状態の厳密一致を確認する。学習レシピやprecisionを変えて動かす方法は含めない。
本番は共有queue resource allとsystemdのライフサイクルを使い、500周期保存・固定validation・
独立readback・全snapshot保持を接続してから明示的に起動する。
