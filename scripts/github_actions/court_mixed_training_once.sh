#!/usr/bin/env bash
# One-shot owner-only driver used by a temporary self-hosted Actions workflow.

set -euo pipefail

readonly HOST_REPO="/home/kamimura/projects/tennis-lab"
readonly HOST_PY="$HOST_REPO/.venv/bin/python"
readonly STATE_ROOT="/var/lib/tennis-lab-actions"
readonly QUEUE="/opt/tennis-lab-actions/bin/training_queue.sh"
readonly DENSE_NAME="court_mixed_hierarchical_dense_only_v3_b8_a16_e100"
readonly POSE_NAME="court_mixed_hierarchical_pose_consistency_v3_b8_a16_e100"
readonly DENSE_ID="run-court-mixed-hierarchical-dense-only-v3-b8-a16-e100"
readonly POSE_ID="run-court-mixed-hierarchical-pose-consistency-v3-b8-a16-e100"
readonly GROUP_ID="group-court-mixed-hierarchical-v3-pose-consistency"

: "${GITHUB_REPOSITORY:?}"
: "${GITHUB_REPOSITORY_OWNER:?}"
: "${GITHUB_ACTOR:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_SHA:?}"
: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_REF_NAME:?}"

if [[ "$GITHUB_REPOSITORY" != "Motoki0705/tennis-lab" ]]; then
  echo "Unexpected repository: $GITHUB_REPOSITORY" >&2
  exit 1
fi
if [[ "$GITHUB_ACTOR" != "$GITHUB_REPOSITORY_OWNER" ]]; then
  echo "Only the repository owner may run this one-shot workflow." >&2
  exit 1
fi
if [[ ! -x "$HOST_PY" || ! -x "$QUEUE" ]]; then
  echo "Required host runtime is unavailable." >&2
  exit 1
fi
systemctl is-active --quiet tennis-lab-training-queue.service

common_overrides="model=hierarchical data/processing=all data/augmentation=pose_safe data.source.keypoint_court_scope=target_court data.batch_size=8 data.num_workers=8 paths.data_root=$HOST_REPO/data paths.external_asset_root=$HOST_REPO/third_party paths.cache_root=$HOST_REPO/.cache training.compile.enabled=false training.trainer.max_epochs=100 training.trainer.accumulate_grad_batches=16 training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false training.early_stopping.enabled=false training.qualitative_logging.enabled=false run.seed=42 run.test_after_fit=true"
readonly DENSE_COMMAND="CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $HOST_PY -m src.tasks.court_detection.scripts.train_mixed $common_overrides loss.pose.enabled=false loss.pose.translation_weight=0.0 loss.pose.rotation_weight=0.0 loss.pose.focal_weight=0.0 loss.consistency.enabled=false loss.consistency.weight=0.0 loss.consistency.cheirality_weight=0.0 loss.consistency.warmup_fraction=0.0 run.output_dir=court_detection/mixed-source/hierarchical-dense-only-v3-b8-a16-e100"
readonly POSE_COMMAND="CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $HOST_PY -m src.tasks.court_detection.scripts.train_mixed $common_overrides loss.pose.enabled=true loss.pose.translation_weight=1.0 loss.pose.rotation_weight=1.0 loss.pose.focal_weight=1.0 loss.consistency.enabled=true loss.consistency.weight=1.0 loss.consistency.cheirality_weight=0.0 loss.consistency.warmup_fraction=0.0 run.output_dir=court_detection/mixed-source/hierarchical-pose-consistency-v3-b8-a16-e100"

verify_dataset() {
  "$HOST_PY" - <<'PY'
import json
from pathlib import Path

path = Path("/var/lib/tennis-lab-actions/assets/data/synthetic_data_generation/scenes/B00/datasets/court/dataset.json")
payload = json.loads(path.read_text(encoding="utf-8"))
accepted = len(payload.get("samples", []))
rejected = len(payload.get("rejected_samples", []))
proposals = accepted + rejected
assert payload.get("schema") == "canonical_court_dataset_v3", payload.get("schema")
assert payload.get("status") == "completed", payload.get("status")
assert payload.get("seed") == 695, payload.get("seed")
assert payload.get("profile") == "b00-production", payload.get("profile")
assert (proposals, accepted, rejected) == (3336, 3293, 43), (proposals, accepted, rejected)
print(f"B00 V3 verified: proposals={proposals} accepted={accepted} rejected={rejected}")
PY
}

run_checks() {
  cd "$GITHUB_WORKSPACE"
  "$HOST_PY" -m pytest -q \
    tests/unit/tasks/court_detection/test_mixed_training.py \
    tests/unit/synthetic_data_generation/pipeline/test_court_rerun_authority.py \
    tests/unit/synthetic_data_generation/pipeline/test_run_manifest_portability.py
  "$HOST_PY" -m ruff check \
    src/tasks/court_detection/data/mixed.py \
    src/tasks/court_detection/model_io/mixed_adapter.py \
    src/tasks/court_detection/scripts/train_mixed.py \
    src/tasks/court_detection/training/lightning_module_mixed.py \
    src/tasks/court_detection/training/runner_mixed.py \
    tests/unit/tasks/court_detection/test_mixed_training.py \
    tests/unit/synthetic_data_generation/pipeline/test_court_rerun_authority.py \
    tests/unit/synthetic_data_generation/pipeline/test_run_manifest_portability.py
}

materialize_targets() {
  cd "$GITHUB_WORKSPACE"
  "$HOST_PY" -m src.tasks.court_detection.scripts.materialize_targets \
    data/source=synthetic_court \
    data.source.keypoint_court_scope=target_court \
    data/processing=seg_line \
    paths.data_root="$HOST_REPO/data" \
    paths.external_asset_root="$HOST_REPO/third_party"
  "$HOST_PY" -m src.tasks.court_detection.scripts.materialize_targets \
    data/source=tennis_court_detector \
    data/processing=seg_line \
    paths.data_root="$HOST_REPO/data" \
    paths.external_asset_root="$HOST_REPO/third_party"
}

preflight_configs() {
  cd "$GITHUB_WORKSPACE"
  local -a common=(
    model=hierarchical
    data/processing=all
    data/augmentation=pose_safe
    data.source.keypoint_court_scope=target_court
    data.batch_size=8
    data.num_workers=0
    paths.data_root="$HOST_REPO/data"
    paths.external_asset_root="$HOST_REPO/third_party"
    paths.cache_root="$HOST_REPO/.cache"
    training.compile.enabled=false
    run.dry_run=true
    run.test_after_fit=false
  )
  "$HOST_PY" -m src.tasks.court_detection.scripts.train_mixed \
    "${common[@]}" \
    loss.pose.enabled=false \
    loss.pose.translation_weight=0.0 \
    loss.pose.rotation_weight=0.0 \
    loss.pose.focal_weight=0.0 \
    loss.consistency.enabled=false \
    loss.consistency.weight=0.0
  "$HOST_PY" -m src.tasks.court_detection.scripts.train_mixed \
    "${common[@]}" \
    loss.pose.enabled=true \
    loss.pose.translation_weight=1.0 \
    loss.pose.rotation_weight=1.0 \
    loss.pose.focal_weight=1.0 \
    loss.consistency.enabled=true \
    loss.consistency.weight=1.0
}

create_persistent_checkout() {
  local run_root="$STATE_ROOT/runs/${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
  RUN_REPOSITORY="$run_root/repository"
  readonly RUN_REPOSITORY
  if [[ -e "$run_root" ]]; then
    echo "Persistent run directory already exists: $run_root" >&2
    exit 1
  fi
  install -d -m 0700 "$run_root"
  git clone --no-local --no-checkout "$GITHUB_WORKSPACE" "$RUN_REPOSITORY"
  git -C "$RUN_REPOSITORY" remote set-url origin "https://github.com/$GITHUB_REPOSITORY.git"
  git -C "$RUN_REPOSITORY" checkout --detach "$GITHUB_SHA"
  git -C "$RUN_REPOSITORY" submodule update --init --recursive --depth 1
  [[ "$(git -C "$RUN_REPOSITORY" rev-parse HEAD)" == "$GITHUB_SHA" ]]
  for asset_name in data ckpt; do
    local asset_source="$STATE_ROOT/assets/$asset_name"
    local asset_target="$RUN_REPOSITORY/$asset_name"
    [[ -d "$asset_source" && ! -e "$asset_target" && ! -L "$asset_target" ]]
    ln -s "$asset_source" "$asset_target"
  done
}

add_job() {
  local name="$1"
  local command="$2"
  local output
  output="$(
    cd "$RUN_REPOSITORY"
    TRAINING_QUEUE_DIR="$STATE_ROOT/training-queue" \
      "$QUEUE" add "$command" \
        --name "$name" \
        --provider github-actions \
        --session "$GITHUB_RUN_ID" \
        --prune-ckpt
  )"
  printf '%s\n' "$output" >&2
  local job="${output#queued: }"
  [[ "$job" != "$output" && "$job" == *.job ]]
  printf '%s' "$job"
}

wait_job() {
  local job="$1"
  local queue_dir="$STATE_ROOT/training-queue"
  local stem="${job%.job}"
  if [[ ! -f "$queue_dir/done/$job" && ! -f "$queue_dir/failed/$job" ]]; then
    local pattern="done: ${job}|FAILED.*${job}"
    set +o pipefail
    tail -n 20000 -F "$queue_dir/worker.log" | grep -m1 -E "$pattern"
    local wait_status=$?
    set -o pipefail
    [[ "$wait_status" -eq 0 ]]
  fi
  if [[ -f "$queue_dir/failed/$job" ]]; then
    echo "Training failed: $job" >&2
    cat "$queue_dir/logs/${stem}.log" >&2
    return 1
  fi
  [[ -f "$queue_dir/done/$job" ]]
  echo "Training completed: $job"
  tail -n 160 "$queue_dir/logs/${stem}.log"
}

write_knowledge_findings() {
  cd "$RUN_REPOSITORY"
  "$HOST_PY" - <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

scripts = Path(".agents/skills/knowledge-control/scripts").resolve()
sys.path.insert(0, str(scripts))
from kg_lib import dump_frontmatter, parse_node

nodes = Path("knowledge/nodes")
dense_id = "run-court-mixed-hierarchical-dense-only-v3-b8-a16-e100"
pose_id = "run-court-mixed-hierarchical-pose-consistency-v3-b8-a16-e100"
group_id = "group-court-mixed-hierarchical-v3-pose-consistency"


def fmt(value: object) -> str:
    return f"{float(value):.6g}" if isinstance(value, (int, float)) else str(value)


def metrics_table(metrics: dict[str, object]) -> str:
    if not metrics:
        return "テストメトリクスは保存されていない。"
    rows = ["| Metric | Value |", "|---|---:|"]
    rows.extend(f"| `{key}` | {fmt(value)} |" for key, value in sorted(metrics.items()))
    return "\n".join(rows)


def comparison_table(left: dict[str, object], right: dict[str, object]) -> str:
    rows = ["| Metric | Dense only | Pose + consistency | Δ (後者−前者) |", "|---|---:|---:|---:|"]
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            rows.append(f"| `{key}` | {fmt(a)} | {fmt(b)} | {fmt(float(b) - float(a))} |")
    return "\n".join(rows) if len(rows) > 2 else "共通する数値メトリクスはない。"


def rewrite(node_id: str, *, title: str, pose: bool, relations: list[dict[str, str]], body: str) -> dict[str, object]:
    path = nodes / f"{node_id}.md"
    node = parse_node(path)
    meta = node.meta
    meta["title"] = title
    meta["provider"] = "codex"
    meta["config"] = {
        "model": "hierarchical (DINOv3 + transformer encoder + DPT)",
        "data": "Synthetic Court V3 + TennisCourtDetector, 4+4 samples per batch",
        "processing": "kp + seg + line",
        "augmentation": "pose_safe",
        "batch": "micro 8, accumulation 16, effective 128",
        "epochs": 100,
        "seed": 42,
        "pose": pose,
        "consistency": pose,
        "pose_scope": "synthetic_court only" if pose else "disabled",
    }
    meta["parents"] = []
    meta["relations"] = relations
    meta["tags"] = [
        "court-detection",
        "mixed-source",
        "synthetic-court-v3",
        "tennis-court-detector",
        "hierarchical",
        "effective-batch-128",
        *(["pose", "consistency", "synthetic-only-supervision"] if pose else ["dense-only"]),
    ]
    path.write_text(f"---\n{dump_frontmatter(meta)}---\n\n{body.strip()}\n", encoding="utf-8")
    return dict(meta.get("metrics") or {})


dense_path = nodes / f"{dense_id}.md"
pose_path = nodes / f"{pose_id}.md"
dense_metrics = dict(parse_node(dense_path).meta.get("metrics") or {})
pose_metrics = dict(parse_node(pose_path).meta.get("metrics") or {})
compare = comparison_table(dense_metrics, pose_metrics)

dense_body = f"""
## 考察 / Findings

### 要約
Synthetic Court V3とTennisCourtDetectorを各batchで4件ずつ混合し、KP/SEG/LINEのみで100 epoch学習した比較基準runである。poseとconsistencyは明示的に無効化した。

### アーキテクチャ詳細
`hierarchical`のDINOv3 encoder、spatial transformer encoder、DPT decoderを使用した。両sourceへ同じ`pose_safe`変換を適用し、micro-batch 8、gradient accumulation 16でeffective batch 128とした。

### メトリクスの解釈
{metrics_table(dense_metrics)}

test splitを持たないTennisCourtDetectorをvalidationの代替として流用していないため、test値はSynthetic Court V3の明示的test splitを中心に読む。

### アーキテクチャ⇄メトリクスの因果考察
このrun単独から混合学習の因果効果は断定しない。後続runとdata、augmentation、seed、batch、epochを一致させ、追加objectiveの影響を比較する基準とする。

### 既存実験との比較
今回のmatched comparisonにおけるbaselineであり、直接のparent runは設定していない。

### 次に有効な実験
pose+consistencyをsynthetic sampleだけに適用したmatched runと共通dense metricを比較する。
"""
rewrite(
    dense_id,
    title="Court mixed-source hierarchical dense-only (V3, EB128)",
    pose=False,
    relations=[],
    body=dense_body,
)

pose_body = f"""
## 考察 / Findings

### 要約
dense-only基準と同じmixed batchを使い、poseおよびKP–pose consistencyをSynthetic Court V3の4 sampleだけへ適用した。TennisCourtDetector sampleには両lossの教師もgradientも与えていない。

### アーキテクチャ詳細
dense branchは8 sample全体でKP/SEG/LINEを学習し、`pose_supervision_mask`でsynthetic sampleだけを抽出してtranslation、rotation、log-focal、consistencyを計算する。その他の条件はbaselineと一致する。

### メトリクスの解釈
{metrics_table(pose_metrics)}

共通metric差分は以下。Δはpose+consistencyからdense-onlyを引いた値であり、改善方向は各metric定義に従って判断する。

{compare}

### アーキテクチャ⇄メトリクスの因果考察
意図的な差はsynthetic限定pose/consistency objectiveであるため、共通metric差は追加objectiveに関連する観測として扱える。ただし単一seedであり、分散を超える効果かは未確定である。

### 既存実験との比較
`{dense_id}`とのmatched comparisonである。TennisCourtDetector側pose outputのloss gradientが0になることは単体テストでも検証済み。

### 次に有効な実験
追加seedで再現性を測り、consistency weightとwarmupの感度を評価する。
"""
rewrite(
    pose_id,
    title="Court mixed-source hierarchical synthetic-only pose+consistency (V3, EB128)",
    pose=True,
    relations=[{"to": dense_id, "rel": "compares"}],
    body=pose_body,
)

group_meta = {
    "id": group_id,
    "type": "group",
    "title": "Court mixed-source hierarchical: pose/consistency比較 (V3)",
    "members": [dense_id, pose_id],
    "parents": [],
    "tags": ["court-detection", "mixed-source", "synthetic-court-v3", "pose", "consistency", "ablation"],
}
group_body = f"""
## まとめ

Synthetic Court V3とTennisCourtDetectorを同一batch内で4:4に固定したmatched comparisonである。`hierarchical`、`pose_safe`、seed 42、effective batch 128、100 epochを共有し、synthetic限定pose/consistency objectiveの有無だけを比較する。

{compare}

各metricの定義と収束挙動は個別run nodeを正本とする。単一seedのため差分の再現性は追加seedで確認する必要がある。
"""
(nodes / f"{group_id}.md").write_text(
    f"---\n{dump_frontmatter(group_meta)}---\n\n{group_body.strip()}\n",
    encoding="utf-8",
)
PY
}

register_knowledge() {
  local queue_dir="$STATE_ROOT/training-queue"
  cd "$RUN_REPOSITORY"
  ln -sfn "$queue_dir" .training_queue
  local skill=".agents/skills/knowledge-control/scripts"
  "$HOST_PY" "$skill/kg_register.py" "$DENSE_NAME" --provider codex
  "$HOST_PY" "$skill/kg_register.py" "$POSE_NAME" --provider codex
  "$HOST_PY" "$skill/kg_curves.py" "$DENSE_ID" || true
  "$HOST_PY" "$skill/kg_curves.py" "$POSE_ID" || true
  write_knowledge_findings
  "$HOST_PY" "$skill/kg_validate.py"
}

commit_results() {
  cp -a "$RUN_REPOSITORY/knowledge/." "$GITHUB_WORKSPACE/knowledge/"
  cd "$GITHUB_WORKSPACE"
  rm -f \
    .github/workflows/court-mixed-training-once.yml \
    .github/workflows/court-mixed-training-once-v2.yml \
    scripts/github_actions/court_mixed_training_once.sh
  git config user.name "github-actions[bot]"
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git add -A knowledge \
    .github/workflows/court-mixed-training-once.yml \
    .github/workflows/court-mixed-training-once-v2.yml \
    scripts/github_actions/court_mixed_training_once.sh
  git status --short
  git commit -m "exp(court): register mixed-source training results"
  git push origin "HEAD:${GITHUB_REF_NAME}"
}

verify_dataset
run_checks
materialize_targets
preflight_configs
create_persistent_checkout
FIRST_JOB="$(add_job "$DENSE_NAME" "$DENSE_COMMAND")"
readonly FIRST_JOB
SECOND_JOB="$(add_job "$POSE_NAME" "$POSE_COMMAND")"
readonly SECOND_JOB
TRAINING_QUEUE_DIR="$STATE_ROOT/training-queue" "$QUEUE" status
wait_job "$FIRST_JOB"
wait_job "$SECOND_JOB"
register_knowledge
commit_results

{
  echo "### Court mixed-source experiments completed"
  echo
  echo "- Dense-only job: \`$FIRST_JOB\`"
  echo "- Pose+consistency job: \`$SECOND_JOB\`"
  echo "- Knowledge nodes: \`$DENSE_ID\`, \`$POSE_ID\`, \`$GROUP_ID\`"
} >> "${GITHUB_STEP_SUMMARY:?}"
