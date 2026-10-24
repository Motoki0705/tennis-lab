set -euo pipefail
export PATH="$(uv tool dir --bin):$PATH"
export PYTHONPATH=/content/tennis-lab
P=.venv/bin/python
T=outputs/synthetic_data_generation/profile/court-scale-pilot/scripts/timed_command.py
E=outputs/synthetic_data_generation/profile/court-scale-pilot/evidence
A=data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified
mkdir -p "$E"
"$P" "$T" --receipt "$E/asset-download-timing.json" -- "$P" .agents/skills/tennis-drive/scripts/drive.py download "$A" "$A" --verify
"$P" "$T" --receipt "$E/asset-verify-timing.json" -- "$P" -m src.synthetic_data_generation.scripts.court_scene_assets verify --root "$A" --manifest-sha256 a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b
"$P" "$T" --receipt "$E/real-download-timing.json" -- "$P" .agents/skills/tennis-drive/scripts/drive.py download data/court_detection/tennis_court_detector-v1 data/court_detection/tennis_court_detector-v1 --verify
"$P" "$T" --receipt "$E/backbone-download-timing.json" -- "$P" .agents/skills/tennis-drive/scripts/drive.py download ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --verify
"$P" "$T" --receipt "$E/real-split-timing.json" -- "$P" -m src.tasks.court_detection.scripts.freeze_real_evaluation --root data/court_detection/tennis_court_detector-v1 --output outputs/court_detection/evaluate/real-video-split-v1/s42 --version real-video-v1 --seed 42
"$P" - <<'PY'
import json
from pathlib import Path
current=json.loads(Path("outputs/court_detection/evaluate/real-video-split-v1/s42/registry.json").read_text())
previous=json.loads(Path("knowledge/runs/run-court-data-scaling-smoke-s42/real-evaluation-registry.json").read_text())
assert current==previous
print(json.dumps({"registry_matches_prior":True,"summary":current["summary"],"video_overlap":current["video_overlap"]},indent=2))
PY

for PILOT_PHASE in plan interrupt resume fetch; do
  bash -s "$PILOT_PHASE" <<'PILOT_GENERATION_PHASE'
set -euo pipefail
export PATH="$(uv tool dir --bin):$PATH"
export PYTHONPATH=/content/tennis-lab
P=.venv/bin/python
T=outputs/synthetic_data_generation/profile/court-scale-pilot/scripts/timed_command.py
E=outputs/synthetic_data_generation/profile/court-scale-pilot/evidence
A=data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified
FIRST=tmp/court-scale-pilot-generated
RESUMED=tmp/court-scale-pilot-resumed
FETCHED=data/synthetic_data_generation/court-scale/pilot-l4-20261023-dc5c253
DRIVE=data/synthetic_data_generation/court-scale/pilot-l4-20261023-dc5c253
case "$1" in
plan)
  "$P" "$T" --receipt "$E/plan-timing.json" -- "$P" -m src.synthetic_data_generation.scripts.generate_court_scale plan \
    --scenes-root "$A/scenes" --scenes B00 B01 B02 B03 --asset-manifest "$A/assets.json" \
    --asset-manifest-sha256 a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b \
    --production-recipe src/synthetic_data_generation/configs/dataset/court/scale_stratified_counter_production_v1.json \
    --output-root "$FIRST" --generation-seed 42 --trajectory-count 24 --density-level 0
  "$P" - <<'PY'
import hashlib,json,math
from pathlib import Path
from src.synthetic_data_generation.dataset.court.scale.publication import iter_plan_groups
root=Path("tmp/court-scale-pilot-generated")
request=Path(json.loads((root/"planned-requests.json").read_text())["requests"][0])
sha=hashlib.sha256(request.read_bytes()).hexdigest()
assert request.stem==sha
frozen=json.loads(request.read_text())
scenes=[]
for entry in frozen["scenes"]:
    plan=json.loads((root/entry["plan_path"]).read_text())
    groups=list(iter_plan_groups(root,plan))
    scenes.append({"scene_id":entry["scene_id"],"trajectory_count":len(groups),"candidates":sum(len(g["samples"]) for g in groups),"geometry_pass":sum(not s["reasons"] for g in groups for s in g["samples"]),"caches":sum(math.ceil(len(g["samples"])/64) for g in groups),"base_counts":[g["arc_map"]["base_count"] for g in groups]})
value={"request_sha256":sha,"request_path":str(request),"scenes":scenes}
target=Path("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/request.json")
temporary=target.with_suffix(".partial");temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(target)
print(json.dumps(value,indent=2))
PY
  ;;
interrupt)
  REQUEST=$("$P" -c 'import json;print(json.load(open("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/request.json"))["request_path"])')
  "$P" "$T" --receipt "$E/interrupt-timing.json" --sample-gpu -- "$P" -m src.synthetic_data_generation.scripts.generate_court_scale render-colab \
    --output-root "$FIRST" --request "$REQUEST" --drive-path "$DRIVE" --stop-after-caches 1
  "$P" - <<'PY'
import json
from pathlib import Path
root=Path("tmp/court-scale-pilot-generated")
attempts=list((root/"attempts").glob("*.json"));assert len(attempts)==1
attempt=json.loads(attempts[0].read_text())
assert attempt["outcome"]=="interrupted" and len(attempt["caches"])==1
assert attempt["caches"][0]["drive_acknowledged"]
assert not list(root.glob("*.dataset.json"))
preserved=Path("tmp/court-scale-pilot-interrupted")
assert not preserved.exists() and not Path("tmp/court-scale-pilot-resumed").exists()
assert not Path("data/synthetic_data_generation/court-scale/pilot-l4-20261023-dc5c253").exists()
root.rename(preserved)
e=Path("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence")
(e/"interrupted-attempt.partial").write_text(json.dumps(attempt,indent=2)+"\n")
(e/"interrupted-attempt.partial").replace(e/"interrupted-attempt.json")
print(json.dumps({"original_root_absent":not root.exists(),"preserved_root":str(preserved),"acknowledged_cache":attempt["caches"][0]["cache_key"]},indent=2))
PY
  ;;
resume|fetch)
  SHA=$("$P" -c 'import json;print(json.load(open("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/request.json"))["request_sha256"])')
  test ! -e "$FIRST"
  if [ "$1" = resume ]; then
    test ! -e "$RESUMED"
    "$P" "$T" --receipt "$E/resume-timing.json" --sample-gpu -- "$P" -m src.synthetic_data_generation.scripts.generate_court_scale render-colab \
      --output-root "$RESUMED" --request-sha256 "$SHA" --drive-path "$DRIVE"
  else
    test ! -e "$FETCHED"
    "$P" "$T" --receipt "$E/fetch-timing.json" -- "$P" -m src.synthetic_data_generation.scripts.generate_court_scale fetch \
      --output-root "$FETCHED" --request-sha256 "$SHA" --drive-path "$DRIVE"
  fi
  ;;
*) exit 2 ;;
esac

PILOT_GENERATION_PHASE
done
P=.venv/bin/python
T=outputs/synthetic_data_generation/profile/court-scale-pilot/scripts/timed_command.py
E=outputs/synthetic_data_generation/profile/court-scale-pilot/evidence
FETCHED=data/synthetic_data_generation/court-scale/pilot-l4-20261023-dc5c253
SHA=$("$P" -c 'import json;print(json.load(open("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/request.json"))["request_sha256"])')
VERSION=$("$P" -c 'import json,sys;print(json.load(open(sys.argv[1]))["dataset_version_path"])' "$FETCHED/completed-requests/$SHA.json")
"$P" "$T" --receipt "$E/framing-timing.json" -- "$P" outputs/synthetic_data_generation/profile/court-scale-pilot/scripts/framing.py --real-root data/court_detection/tennis_court_detector-v1 --registry outputs/court_detection/evaluate/real-video-split-v1/s42/registry.json --synthetic-version "$FETCHED/$VERSION" --output /content/court-pilot-framing-completed
mv /content/court-pilot-framing-completed outputs/synthetic_data_generation/profile/court-scale-pilot/framing
"$P" "$T" --receipt "$E/review-timing.json" -- "$P" outputs/synthetic_data_generation/profile/court-scale-pilot/scripts/review_generation.py --version "$FETCHED/$VERSION" --output /content/court-pilot-review-completed
mv /content/court-pilot-review-completed outputs/synthetic_data_generation/profile/court-scale-pilot/review
"$P" - <<'PY'
import hashlib,json,tempfile
from pathlib import Path
from src.synthetic_data_generation.dataset.court.scale.publication import load_dataset_version
from src.utils.data.image_record_store import file_sha256
E=Path("outputs/synthetic_data_generation/profile/court-scale-pilot/evidence")
resumed=Path("tmp/court-scale-pilot-resumed")
fetched=Path("data/synthetic_data_generation/court-scale/pilot-l4-20261023-dc5c253")
sha=json.loads((E/"request.json").read_text())["request_sha256"]
name=json.loads((fetched/f"completed-requests/{sha}.json").read_text())["dataset_version_path"]
version=load_dataset_version(fetched/name,verify_images=True)
assert version==load_dataset_version(resumed/name,verify_images=True)
assert all(s["trajectory_count"]==24 and s["density_level"]==0 for s in version["scenes"])
initial=json.loads((E/"interrupted-attempt.json").read_text())
paths=list((resumed/"attempts").glob("*.json"));assert len(paths)==1
attempt=json.loads(paths[0].read_text())
assert initial["outcome"]=="interrupted" and attempt["outcome"]=="completed"
first=initial["caches"][0]["cache_key"]
events=attempt["caches"]
assert len(events)==len({x["cache_key"] for x in events})==sum(len(s["render_caches"]) for s in version["scenes"])
restored=[x for x in events if x["reused"] or x["restored_from_drive"]]
assert restored==[{"cache_key":first,"reused":True,"restored_from_drive":True,"drive_acknowledged":True}]
assert all(x["drive_acknowledged"] for x in events)
assert not Path("tmp/court-scale-pilot-generated").exists()
for obj in (initial,attempt):
    release=obj["renderer_release"]
    assert release["renderer_context_released"] and release["compute_processes_after"]==[]
owners={"requests","plans","curves","assets","stores","cache-metrics","memberships","reports","completed-requests"}
def inventory(root):
    return {p.relative_to(root).as_posix():file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and (p.relative_to(root).parts[0] in owners or p.name.endswith(".dataset.json"))}
original=inventory(resumed);recovered=inventory(fetched);assert original==recovered
result={"schema":"court_pilot_recovery_verification_v1","version_path":name,"version_sha256":file_sha256(fetched/name),"request_sha256":sha,"inventory_files":len(recovered),"inventory_sha256":hashlib.sha256(json.dumps(recovered,sort_keys=True).encode()).hexdigest(),"caches":len(events),"restored_caches":len(restored),"new_caches":len(events)-len(restored),"accepted_images":sum(s["counts"]["qa_accepted"] for s in version["scenes"]),"strict_fetch_equal":True,"renderer_released":True}
copies=[(fetched/name,"dataset-version.json"),(fetched/f"reports/{sha}.json","generation-report.json"),(paths[0],"resumed-attempt.json"),(fetched/version["request_path"],"frozen-request.json")]
with tempfile.TemporaryDirectory(dir="/content",prefix="pilot-evidence-") as temporary:
    for i,(source,target) in enumerate(copies):
        t=Path(temporary)/str(i);t.write_bytes(source.read_bytes());assert not (E/target).exists();t.replace(E/target)
    p=Path(temporary)/"verification.json";p.write_text(json.dumps(result,indent=2)+"\n");p.replace(E/"verification.json")
    metrics=[json.loads(p.read_text()) for p in sorted((fetched/"cache-metrics").glob("*.json"))]
    p=Path(temporary)/"cache-metrics.json";p.write_text(json.dumps(metrics,indent=2)+"\n");p.replace(E/"cache-metrics.json")
print(json.dumps(result,indent=2))
PY
"$P" "$T" --receipt "$E/drive-dataset-inspect-timing.json" -- "$P" .agents/skills/tennis-drive/scripts/drive.py inspect "$FETCHED"
