# BLCS loss ablation

位置教師にclean再投影、速度教師、飛行区間内の位置／速度rollout、任意の物理headを段階的に加え、同じ更新数で比較する。checkpointの選択にはvalidationの `trajectory_rmse_m` だけを使う。

| module | 責務 |
|---|---|
| `progressive_losses.py` | 6条件の明示的な損失設定、固定更新数での学習と選定receipt |
| `evaluation.py` | checkpointと幾何・GT摂動対照の全長評価、予測保存、保存指標の再計算 |
| `real_observations.py` | 校正済み多視点観測の再投影・物理診断と保存値の再計算 |
| `verification.py` | 保存値と再計算の値・支持数・未定義状態の照合 |
| `cli_paths.py` | 公開CLIの入出力path検証 |

実行入口はユーザー指定の `src/tasks/ball_detection/scripts/` に集約する。以下のGPUコマンドは[training queue](../../../../.agents/skills/training-queue/SKILL.md)経由で実行する。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train_blcs_ablation \
  --arm rollout_velocity --seed 42 --steps 4000 \
  --data-root /absolute/data --scene-dir blcs/single_object \
  --output blcs/ablation/rollout-velocity-s42
.venv/bin/python -m src.tasks.ball_detection.scripts.evaluate_blcs_ablation \
  --data-root /absolute/data --checkpoint /absolute/model.ckpt \
  --output /absolute/outputs/evaluation
```

`--arm` は `position` → `reprojection` → `velocity` → `rollout_position` → `rollout_velocity`（A）→ `parameters`（B）。Bはheadと補助損失を一群として追加するため、個々のB損失を切り分けた比較ではない。通常学習の標準objectiveはこの入口と独立で、新しい物理損失は `loss=physics` で明示的に有効化する。

評価は全有効frameの3D軌道RMSEを主軸とし、速度head・位置差分の速度／加速度誤差、イベントを跨がない積分残差、clean再投影、支持数と欠損・イベント近傍等の層別を保存する。支持数0はJSONでnull。固定128frame窓を連結するため窓境界の差分誤差に注意する。

`evaluation.recompute(path, include_diagnostics=True)` は保存NPZの予測と積分状態から指標を再計算し、積分器自体は再実行しない。実観測は独立3D GTを持たず、固定2D注釈・保存校正からの診断である。`real_observations.recompute_real(directory)` は保存済みの重力も使う。重力を保存していない旧bundleには `gravity=<実験時の値>` を明示し、不明な値を推測しない。

Issue #1044の[学習前固定プロトコル](../../../../knowledge/runs/group-i1044-progressive-losses-u4000/protocol.md)と[最終比較](../../../../knowledge/nodes/blcs/000068-group-i1044-progressive-losses-u4000.md)はknowledgeに保持する。固定seed追加・10run列挙・一括report生成の専用controllerは常設コードから除去し、元のrunに記録されたcommitとreproで再現する。rawログ・receipt・reproコマンドは書き換えない。既存BLCSの生成・通常学習・UI入口は `src/tasks/blcs/scripts/` が担当する。
