---
id: run-mcp-court-old-service
type: run
title: mcp-court-real-three-batches
provider: codex
session: train-1a3f3c1500e57dfc
date: '2026-09-12'
status: failed
config:
  model: court_hierarchical / DINOv3 + DPT
  loss: court KP loss
  data: existing court images
metrics: {}
repro:
  commit: 615fb03e21a86e7cb517758b7b96cecd4cdffb0e
  branch: HEAD
  remote: https://github.com/Motoki0705/tennis-lab.git
  command: env TENNIS_MCP_REPO_ROOT=/home/kamimura/projects/tennis-lab TENNIS_MCP_STATE_DIR=/home/kamimura/.local/state/tennis-lab-chatgpt-mcp
    TENNIS_MCP_CONTROL_DIR=/home/kamimura/.local/share/tennis-lab-chatgpt-mcp TENNIS_MCP_ORIGIN_URL=https://github.com/Motoki0705/tennis-lab.git
    TENNIS_MCP_DOCKER_IMAGE=nvidia/cuda:13.0.0-base-ubuntu24.04 TENNIS_MCP_UV_PYTHON_ROOT=/home/kamimura/.local/share/uv/python
    /home/kamimura/.local/share/tennis-lab-chatgpt-mcp/venv/bin/python -I -c 'import
    sys; from pathlib import Path; sys.path.insert(0, '"'"'/home/kamimura/.local/share/tennis-lab-chatgpt-mcp/current'"'"');
    from src.automation.chatgpt_mcp.sandbox_exec import run_from_spec; raise SystemExit(run_from_spec(Path(sys.argv[1])))'
    /home/kamimura/.local/state/tennis-lab-chatgpt-mcp/training-specs/train-1a3f3c1500e57dfc.json
  workload_command: knowledge/runs/run-mcp-court-old-service/workload-command.sh
artifacts:
  run_dir: knowledge/runs/run-mcp-court-old-service
  log: knowledge/runs/run-mcp-court-old-service/output.log
parents: []
relations: []
tags:
- mcp
- capability-check
---


## 考察 / Findings

### 要約
サービスの実行パスが旧リリースを指していたため旧CUDAベースイメージが使用され、OpenCVのlibxcb不足で停止した。

### アーキテクチャ詳細
MCPのネットワーク隔離コンテナと既存のGPU queueを利用。実行条件とイメージIDは同じrun bundleのresult.json、実ログはoutput.logに記録した。

### メトリクスの解釈
学習開始前の依存解決失敗であり、GPU計算能力の失敗とは区別する。

### アーキテクチャ⇄メトリクスの因果考察
runtime-versionファイルの更新だけではロード済みコードを表さない。serviceのWorkingDirectory/PYTHONPATHと実コンテナImageを確認する必要がある。

### 既存実験との比較
後続の専用イメージによる検証と比較するための失敗記録。

### 次に有効な実験
ChatGPT接続をRefreshし、モデル向け定義とresource=halfの実呼び出しを確認する。
