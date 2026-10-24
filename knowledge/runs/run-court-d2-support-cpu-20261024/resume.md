# 親がpushした後のL4比較

今回の実行はCPUまで。新規描画0、学習0、Colab session開始0。
`render_probe.py` はCPUで選択と全192カメラの再構成まで検査済みで、L4実行・画質は未検証。
本番recipe、採用bundle、前回pilot dataset、既存Drive出力を変更しない。

## 開始前

1. 親がこのbranchの新commitをpushし、正確なcommitを指定する。mergeされた版でもよいが、
   `render-selection.json` とscript一式がそのcommitに含まれることを確認する。
2. [tennis-colab](../../../.agents/skills/tennis-colab/SKILL.md) と
   [tennis-drive](../../../.agents/skills/tennis-drive/SKILL.md) を読み直す。
3. 一意なsessionと新しいoutput rootを選ぶ。以下はローカルで実行する。
   `PUSHED_COMMIT` は親が確認したcommitに置き換える。設定の中身を表示しない。

```bash
export RCLONE_CONFIG="$HOME/.config/tennis-lab/rclone-drive.conf"
COURT_PROBE_REF=PUSHED_COMMIT
COURT_PROBE_SESSION=court-d2-probe-l4-$(date -u +%Y%m%dT%H%M%SZ)
COURT_PROBE_OUTPUT=outputs/synthetic_data_generation/profile/court-d2-support-l4/$COURT_PROBE_SESSION
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py doctor --remote
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py start \
  --session "$COURT_PROBE_SESSION" --ref "$COURT_PROBE_REF" --dry-run
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py start \
  --session "$COURT_PROBE_SESSION" --ref "$COURT_PROBE_REF"
```

要求・観測ともL4、HEAD一致を確認する。他sessionを変更しない。
以下のsetupを `colab.py exec --session ... --job-id setup --persist ... -- bash -lc ...` で提出する。
実際のjob requestも新runへ保存する。

```bash
set -euo pipefail
export PATH="$(uv tool dir --bin):$PATH"
uv sync --locked
.venv/bin/python -m spin setup-nht --render-only
COURT_PROBE_ASSETS=data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download \
  "$COURT_PROBE_ASSETS" "$COURT_PROBE_ASSETS" --verify
.venv/bin/python -m src.synthetic_data_generation.scripts.court_scene_assets verify \
  --root "$COURT_PROBE_ASSETS" \
  --manifest-sha256 a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b
```

`setup` のterminal・returncode=0・drive_saved_atを確認する。setupの稼働ログを
periodic persist対象へ置かず、終了後のsnapshotを保存する。新VMには既存資産が無い前提。
再試行時は部分取得をinspectし、同じ場所へ盲目的に再downloadしない。

## 192枚だけ描画

ローカルから次を提出する。output rootは新規で、全frame・QA・raw RGB/alpha/depth・JSONを
workerのperiodic persistで新しいDrive pathへ保存する。入力datasetではなく画質診断の実験出力であり、
本番学習用のdataset_versionは発行しない。通常のproduction専用render-colab CLIへ比較requestを渡さない。
このscriptは既存ColabNHTRenderer／evaluate_renderを直接使う。

```bash
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py exec \
  --session "$COURT_PROBE_SESSION" --job-id render-probe \
  --persist "$COURT_PROBE_OUTPUT" -- \
  bash -lc 'set -euo pipefail; export PATH="$(uv tool dir --bin):$PATH"; export PYTHONPATH="$PWD"; .venv/bin/python knowledge/runs/run-court-d2-support-cpu-20261024/render_probe.py --selection knowledge/runs/run-court-d2-support-cpu-20261024/render-selection.json --asset-root data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified --output "$1"' \
  _ "$COURT_PROBE_OUTPUT"
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py status \
  --session "$COURT_PROBE_SESSION" --job-id render-probe
```

完了条件はstateのdraw_attempts=completed_frames=192、outcome=completed、L4のcontext解放、
job returncode=0とDrive保存receipt。提出receiptだけで完了にしない。
D5の対象KP≥4・ROI穴≤1%を適用し、α<0.9は診断のまま。QA拒否RGBも診断として残し、採択入力へは使わない。
カメラ高さやD6、生成・選択・QAのsrcは変更していない。

**失敗時も描画を自動再実行しない。** submitted camera数を描画消費として数える。
VM消失や成否不明なら当192枚全部を消費したと保守的に扱い、新rootで再試行しない。
この段階の総数上限200から残り枠を親が確認してから判断する。

## 回収・報告と終了（失敗時も必須）

- 全16scenarioのreportsを回収し、条件/scene/framingごとのQA、拒否理由、α<0.9を比較する。
  raw層別採択率とN_h/n_hで重み付けした参考母集団率を併記する。新しい除外・補充はしない。
- 原画とoverlay、低視点のぼけ・暗さ・hull外破綻を実見する。幾何上のfullを画質の合格にしない。
- raw保存は本番JPEG-only publicationより大きく、NPZ圧縮・QA保存を含むprobe時間を
  そのまま本番の秒/枚に移植しない。render/QA/archive・worker保存・setupを分けて記録する。
  必要ならこの少数probeからの不確かな再推定と前回モデルによる値を分ける。
- Drive subtreeをinspectし、必要な小さなJSON/PNGだけ取得・verifyする。新node/run dirへ記録する。
  採択率と容量の仮定を更新した後、D2採否と本実験のmax/G/U/seedを親へ返す。
- **例外や失敗の場合も**、稼働jobを調べ、必要ならcancelし、terminalとDrive保存を確認する。
  保存失敗はskillのsave/recover経路で解決する。全jobの保存後、必ず次を実行する。

```bash
.venv/bin/python .agents/skills/tennis-colab/scripts/colab.py stop \
  --session "$COURT_PROBE_SESSION"
```

stop receiptと回収diffを保存する。GPU context解放だけでVM停止済みとは扱わない。
