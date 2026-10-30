set -euo pipefail
export PATH="$(uv tool dir --bin):$PATH"
ASSETS=data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified
ASSET_SHA=a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b
FIRST=tmp/court-scale-production-smoke
RESUMED=tmp/court-scale-resumed-smoke
FETCHED=tmp/court-scale-from-drive-smoke
DRIVE=data/synthetic_data_generation/court-scale/smoke-20261023-6ed3f29e8
EVIDENCE=outputs/synthetic_data_generation/profile/court-scale-smoke/evidence
mkdir -p "$EVIDENCE"
TIMEFORMAT="elapsed_seconds=%R"

case "$1" in
  inputs)
    time .venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download \
      "$ASSETS" "$ASSETS" --verify > /content/.court-scale-input-download.json
    mv /content/.court-scale-input-download.json "$EVIDENCE/input-download.json"
    time .venv/bin/python -m src.synthetic_data_generation.scripts.court_scene_assets verify \
      --root "$ASSETS" --manifest-sha256 "$ASSET_SHA" > /content/.court-scale-input-verify.json
    mv /content/.court-scale-input-verify.json "$EVIDENCE/input-verification.json"
    cat "$EVIDENCE/input-verification.json"
    ;;
  plan)
    time .venv/bin/python -m src.synthetic_data_generation.scripts.generate_court_scale plan \
      --scenes-root "$ASSETS/scenes" --scenes B00 B01 B02 B03 \
      --asset-manifest "$ASSETS/assets.json" --asset-manifest-sha256 "$ASSET_SHA" \
      --production-recipe src/synthetic_data_generation/configs/dataset/court/scale_stratified_counter_production_v1.json \
      --output-root "$FIRST" --generation-seed 42 --trajectory-count 6 --density-level 0
    .venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("tmp/court-scale-production-smoke")
request = Path(json.loads((root / "planned-requests.json").read_text())["requests"][0])
sha = hashlib.sha256(request.read_bytes()).hexdigest()
assert request.stem == sha
evidence = Path("outputs/synthetic_data_generation/profile/court-scale-smoke/evidence")
temporary = evidence / "request.tmp"
temporary.write_text(json.dumps({"request_sha256": sha, "request_path": str(request)}, indent=2) + "\n")
temporary.replace(evidence / "request.json")
print(sha)
PY
    ;;
  interrupt)
    REQUEST=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/synthetic_data_generation/profile/court-scale-smoke/evidence/request.json"))["request_path"])')
    time .venv/bin/python -m src.synthetic_data_generation.scripts.generate_court_scale render-colab \
      --output-root "$FIRST" --request "$REQUEST" --drive-path "$DRIVE" --stop-after-caches 1
    .venv/bin/python - <<'PY'
import json
from pathlib import Path

from src.synthetic_data_generation.dataset.court.scale.durable_storage import SkillDriveFiles

root = Path("tmp/court-scale-production-smoke")
attempts = list((root / "attempts").glob("*.json"))
assert len(attempts) == 1
attempt = json.loads(attempts[0].read_text())
assert attempt["outcome"] == "interrupted"
assert len(attempt["caches"]) == 1 and attempt["caches"][0]["drive_acknowledged"]
assert not list(root.glob("*.dataset.json"))
remote = SkillDriveFiles("data/synthetic_data_generation/court-scale/smoke-20261023-6ed3f29e8")
assert not remote.exists("receipts/completed.json")
assert not Path("tmp/court-scale-resumed-smoke").exists()
assert not Path("tmp/court-scale-from-drive-smoke").exists()
preserved = Path("tmp/court-scale-interrupted-preserved")
assert not preserved.exists()
root.rename(preserved)
evidence = Path("outputs/synthetic_data_generation/profile/court-scale-smoke/evidence")
temporary = evidence / "interrupted-attempt.tmp"
temporary.write_text(json.dumps(attempt, indent=2) + "\n")
temporary.replace(evidence / "interrupted-attempt.json")
proof = {
    "schema": "court_scale_smoke_isolation_v1",
    "original_root": str(root),
    "preserved_root": str(preserved),
    "original_root_absent": not root.exists(),
    "resume_root_absent_before": not Path("tmp/court-scale-resumed-smoke").exists(),
    "fetch_root_absent_before": not Path("tmp/court-scale-from-drive-smoke").exists(),
    "completed_drive_receipt_absent": True,
    "acknowledged_cache_keys": [event["cache_key"] for event in attempt["caches"]],
}
temporary = evidence / "isolation.tmp"
temporary.write_text(json.dumps(proof, indent=2) + "\n")
temporary.replace(evidence / "isolation.json")
print(json.dumps(proof, indent=2))
PY
    ;;
  resume|fetch)
    REQUEST_SHA=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/synthetic_data_generation/profile/court-scale-smoke/evidence/request.json"))["request_sha256"])')
    test ! -e "$FIRST"
    if [ "$1" = resume ]; then
      test ! -e "$RESUMED"
      time .venv/bin/python -m src.synthetic_data_generation.scripts.generate_court_scale render-colab \
        --output-root "$RESUMED" --request-sha256 "$REQUEST_SHA" --drive-path "$DRIVE"
    else
      test ! -e "$FETCHED"
      time .venv/bin/python -m src.synthetic_data_generation.scripts.generate_court_scale fetch \
        --output-root "$FETCHED" --request-sha256 "$REQUEST_SHA" --drive-path "$DRIVE"
    fi
    ;;
  *)
    printf 'Expected inputs, plan, interrupt, resume, or fetch\n' >&2
    exit 2
    ;;
esac
