#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
# Run in this branch's checkout, with run70/73 outputs mounted by storage-tier.
# The evaluator refuses source changes after freeze and refuses cache collisions.
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/prepare_outputs.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/check_inputs.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/diagnose.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/diagnose_scene_quality.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/evaluate.py --tag regression --workers 10
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/evaluate.py --tag meiji4 --dataset meiji_4cam --workers 10
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/api_parity.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/summarize.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/transitions.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/tables.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/diagnose_ambiguity.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/check_mutations.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/render.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/inspect_rgb.py
.venv/bin/python knowledge/runs/run-court-semantic-support-20261117/figures.py
