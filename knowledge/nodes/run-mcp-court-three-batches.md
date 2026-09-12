---
id: run-mcp-court-three-batches
type: run
title: mcp-court-real-three-batches
provider: codex
session: train-e6945c18dd604c54
date: '2026-09-12'
status: done
config:
  model: court_hierarchical / DINOv3 + DPT
  loss: court KP loss
  data: existing court images
metrics:
  optimizer_steps: 3
  finite_gradient_steps: 3
  first_train_loss: 0.2018318474292755
  last_train_loss: 0.20167946815490723
repro:
  commit: 615fb03e21a86e7cb517758b7b96cecd4cdffb0e
  branch: HEAD
  remote: https://github.com/Motoki0705/tennis-lab.git
  command: env TENNIS_MCP_REPO_ROOT=/home/kamimura/projects/tennis-lab TENNIS_MCP_STATE_DIR=/home/kamimura/.local/state/tennis-lab-chatgpt-mcp
    TENNIS_MCP_CONTROL_DIR=/home/kamimura/.local/share/tennis-lab-chatgpt-mcp TENNIS_MCP_ORIGIN_URL=https://github.com/Motoki0705/tennis-lab.git
    TENNIS_MCP_DOCKER_IMAGE=tennis-lab-mcp:runtime-v1 TENNIS_MCP_UV_PYTHON_ROOT=/home/kamimura/.local/share/uv/python
    /home/kamimura/.local/share/tennis-lab-chatgpt-mcp/venv/bin/python -I -c 'import
    sys; from pathlib import Path; sys.path.insert(0, '"'"'/home/kamimura/.local/share/tennis-lab-chatgpt-mcp/current'"'"');
    from src.automation.chatgpt_mcp.sandbox_exec import run_from_spec; raise SystemExit(run_from_spec(Path(sys.argv[1])))'
    /home/kamimura/.local/state/tennis-lab-chatgpt-mcp/training-specs/train-e6945c18dd604c54.json
  workload_command: knowledge/runs/run-mcp-court-three-batches/workload-command.sh
artifacts:
  run_dir: knowledge/runs/run-mcp-court-three-batches
  log: knowledge/runs/run-mcp-court-three-batches/output.log
parents:
- run-mcp-court-old-service
relations: []
tags:
- mcp
- capability-check
---


## 考察 / Findings

### 要約
既存画像とDINOv3事前学習重みを使い、court_detectionで3バッチのforward/backward/optimizer更新が成功。

### アーキテクチャ詳細
MCPのネットワーク隔離コンテナと既存のGPU queueを利用。実行条件とイメージIDは同じrun bundleのresult.json、実ログはoutput.logに記録した。

### メトリクスの解釈
短時間の実行能力検証であり、評価データ上の精度や長時間学習の保証ではない。

### アーキテクチャ⇄メトリクスの因果考察
共有venvを変更せず、実行イメージのOS依存を満たすことで処理が通った。

### 既存実験との比較
旧イメージでの失敗を親ノードとして保持した。

### 次に有効な実験
ChatGPT接続をRefreshし、モデル向け定義とresource=halfの実呼び出しを確認する。
