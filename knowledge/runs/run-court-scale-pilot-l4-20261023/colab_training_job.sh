set -euo pipefail
export PYTHONPATH=/content/tennis-lab
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
P=.venv/bin/python
S=outputs/synthetic_data_generation/profile/court-scale-pilot/scripts
T="$S/timed_command.py"
E=outputs/synthetic_data_generation/profile/court-scale-pilot/evidence
FETCHED=data/synthetic_data_generation/court-scale/pilot-l4-20261023-dc5c253
SHA=$("$P" -c 'import json;print(json.load(open("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/request.json"))["request_sha256"])')
VERSION=$("$P" -c 'import json,sys;print(json.load(open(sys.argv[1]))["dataset_version_path"])' "$FETCHED/completed-requests/$SHA.json")
MANIFEST="synthetic_data_generation/court-scale/pilot-l4-20261023-dc5c253/$VERSION"
RUN=outputs/court_detection/train/court-scale-pilot-l4/s42
"$P" - <<'PY'
import json
from collections import Counter
from pathlib import Path
from src.utils.data.image_record_store import file_sha256
root=Path("data/synthetic_data_generation/court-scale/pilot-l4-20261023-dc5c253")
sha="0e4a507a63382d2df1ab40e1294d943317665f219d2e631b32a8888ab2ea0350"
name=json.loads((root/f"completed-requests/{sha}.json").read_text())["dataset_version_path"]
version=json.loads((root/name).read_text())
rows=[]
for scene in version["scenes"]:
    for entry in scene["render_caches"]:
        path=root/Path(entry["path"]).with_name("qa.json")
        ledger=json.loads(path.read_text());counts=Counter(row["state"] for row in ledger["records"])
        assert not set(counts)-{"accepted","qa_rejected","geometry_rejected"}
        rows.append({"scene_id":scene["scene_id"],"cache_key":ledger["cache_key"],"source_ledger_path":path.relative_to(root).as_posix(),"source_ledger_sha256":file_sha256(path),"candidate_unique":len(ledger["records"]),"geometry_pass":counts["accepted"]+counts["qa_rejected"],"qa_accepted":counts["accepted"],"qa_rejected":counts["qa_rejected"],"geometry_rejected":counts["geometry_rejected"]})
p=Path("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/cache-counters.json")
t=p.with_suffix(".partial");t.write_text(json.dumps({"schema":"court_pilot_cache_counters_v1","caches":rows},indent=2)+"\n");t.replace(p)
PY
A=data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified
for PILOT_PROBE in small dense single base; do
  case "$PILOT_PROBE" in
    small) PILOT_SCENES=(B00 B01 B02 B03); PILOT_GROUPS=6; PILOT_DENSITY=0 ;;
    dense) PILOT_SCENES=(B00 B01 B02 B03); PILOT_GROUPS=24; PILOT_DENSITY=1 ;;
    single) PILOT_SCENES=(B00); PILOT_GROUPS=24; PILOT_DENSITY=0 ;;
    base) PILOT_SCENES=(B00 B01 B02 B03); PILOT_GROUPS=128; PILOT_DENSITY=0 ;;
    *) exit 2 ;;
  esac
  "$P" "$T" --receipt "$E/plan-$PILOT_PROBE-timing.json" -- "$P" -m src.synthetic_data_generation.scripts.generate_court_scale plan \
    --scenes-root "$A/scenes" --scenes "${PILOT_SCENES[@]}" --asset-manifest "$A/assets.json" \
    --asset-manifest-sha256 a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b \
    --production-recipe src/synthetic_data_generation/configs/dataset/court/scale_stratified_counter_production_v1.json \
    --output-root "tmp/court-pilot-plan-cost-$PILOT_PROBE" --generation-seed 42 --trajectory-count "$PILOT_GROUPS" --density-level "$PILOT_DENSITY"
done
"$P" - <<'PY'
import json,math
from collections import Counter
from pathlib import Path
from src.synthetic_data_generation.dataset.court.scale.publication import iter_plan_groups
values=[]
for label in ("small","dense","single","base"):
    root=Path(f"tmp/court-pilot-plan-cost-{label}")
    request=Path(json.loads((root/"planned-requests.json").read_text())["requests"][0])
    frozen=json.loads(request.read_text())
    scenes=[]
    for scene in frozen["scenes"]:
        plan=json.loads((root/scene["plan_path"]).read_text())
        groups=candidates=caches=geometry_pass=0
        for group in iter_plan_groups(root,plan):
            groups+=1;candidates+=len(group["samples"])
            geometry_pass+=sum(not s["reasons"] for s in group["samples"])
            levels=Counter(s["scale"]["birth_level"] for s in group["samples"])
            caches+=sum(math.ceil(n/64) for n in levels.values())
        scenes.append({"scene_id":scene["scene_id"],"groups":groups,"candidates":candidates,"geometry_pass":geometry_pass,"caches":caches})
    t=json.loads(Path(f"outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/plan-{label}-timing.json").read_text())
    values.append({"probe":label,"wall_seconds":t["wall_seconds"],"request_sha256":request.stem,"scene_count":len(scenes),"density_level":frozen["scenes"][0]["density_level"],"scenes":scenes,"rendered":False})
target=Path("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/cpu-plan-probes.json")
temp=target.with_suffix(".partial");temp.write_text(json.dumps({"schema":"court_pilot_cpu_plan_cost_v1","probes":values},indent=2)+"\n");temp.replace(target)
print(json.dumps(values,indent=2))
PY
"$P" "$T" --receipt "$E/train-timing.json" --sample-gpu -- "$P" "$S/profile_training.py" --receipt "$E/training-io.json" -- \
  --config-name train_data_scaling \
  paths.data_root=/content/tennis-lab/data paths.checkpoint_root=/content/tennis-lab/ckpt \
  paths.external_asset_root=/content/tennis-lab/third_party \
  paths.output_root=/content/tennis-lab/outputs paths.artifact_root=/content/tennis-lab/outputs \
  "data.source.manifest=$MANIFEST" \
  mixed.sources.tennis_court_detector.registry=court_detection/evaluate/real-video-split-v1/s42/registry.json \
  data.batch_size=8 mixed.train_batch_counts.synthetic_court=8 data.num_workers=4 \
  training.compile.enabled=true training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false \
  training.trainer.log_every_n_steps=1 training.warmup_steps=4 training.step_schedule.max_steps=32 \
  training.step_schedule.validation_every_n_steps=32 training.step_schedule.checkpoint_every_n_steps=32 \
  training.checkpoint.save_top_k=-1 run.seed=42 \
  run.output_dir=court_detection/train/court-scale-pilot-l4/s42 \
  run.artifact_store.mode=rclone run.artifact_store.remote=gdrive \
  run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/court-scale-pilot-l4/s42 \
  run.artifact_store.sync_interval_seconds=60
"$P" "$S/summarize_training.py" --run-root "$RUN" --evidence-root "$E" --expected-step 32
CHECKPOINT="$RUN/logs/version_0/checkpoints/update-00000032.ckpt"
for PILOT_SPLIT in val test; do
  "$P" "$T" --receipt "$E/evaluate-$PILOT_SPLIT-timing.json" --sample-gpu -- \
    "$P" "$S/evaluate_framing.py" --checkpoint "$CHECKPOINT" --expected-step 32 --split "$PILOT_SPLIT" \
    --assignments outputs/synthetic_data_generation/profile/court-scale-pilot/framing/assignments.jsonl \
    --output "/content/court-pilot-evaluation-$PILOT_SPLIT-completed"
  EVALUATED="outputs/court_detection/evaluate/court-scale-pilot-$PILOT_SPLIT/s42"
  test ! -e "$EVALUATED"
  mkdir -p "$(dirname "$EVALUATED")"
  mv "/content/court-pilot-evaluation-$PILOT_SPLIT-completed" "$EVALUATED"
done
"$P" - <<'PY'
import json,subprocess
from pathlib import Path
r=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True)
assert not r.stdout.strip()
p=Path("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/post-training-release.json")
t=p.with_suffix(".partial");t.write_text(json.dumps({"compute_processes":[],"released_after_training_and_evaluation":True},indent=2)+"\n");t.replace(p)
print("All training/evaluation processes have released the L4.")
PY

