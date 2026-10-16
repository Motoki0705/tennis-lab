# 保存済みlocal-agent campaign

ballの旧`run`・`qa`・`intake` CLIは撤去済み。実行・画像レビュー・採用には
[注釈ワークフロー](../../../tasks/annotation/README.md)を使う。

このpackageには保存済みcampaignの読取・調査用の形式と補助ツールを残す。
旧注釈のschema・検証は[runtime](../runtime/contracts.py)、保存成果物の移行は
[migration](../../../tasks/annotation/migration/README.md)が担当する。
既存のball/player成果物を読めることと、新しいball教師を採用する入口を分ける。

## 残る補助ツール

入口は `.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent`。
`init`以外は`--campaign`に既存campaignの絶対pathを指定する。

- `status` / `efficiency`: 保存済み台帳・sessionから状態、token量、時間を読む。
- `phase2` / `audit`: 旧版と候補の比較資料・確認画像を生成する。これらを新経路の採用証拠にはしない。
- `worker` / `prefetch`: 旧attemptの画像・候補・編集用の補助ツール。新規ballの運用入口は上記ワークフロー。
- `init`: 補助ツール用のcampaign設定を作る。既存campaignを上書きしない。

旧workerの操作仕様は[WORKER.md](WORKER.md)、注釈形式は[PROTOCOL](../resources/PROTOCOL.md)を参照する。
GPU候補計算は[training queue](../../../../.agents/skills/training-queue/SKILL.md)経由に限定する。

## 出力と保管

```text
<annotation_root>/
  _preparation/                         # 入力の正本manifest
  videos/ または done/                  # 対応する保存済み動画
  annotated/
    raw/<sha256>.zip                     # 不変の提出原本
    processing/<sha256>.json             # 判断・出力hash・差し替えの記録
    processed/ball/<clip_id>.json         # 採用済みの正本
    history/ball/<clip_id>/<sha256>.json  # 差し替え前の旧版（campaignとは別保管）
<campaign_dir>/
  campaign.json / control.json / state.json / versions.json
  tasks/<task_id>/attempt_NN/            # task・prompt・annotation・NOTES・result・events
    work/                               # 画像・候補・中間ファイル
  cache/                                # 動画検証・モデル候補・ロック
  qa/                                   # 比較・確認画像
  logs/                                 # 実行イベント・効率集計
```

原本・処理記録・processed・historyは既存成果物として保持する。
状態や画像を再生成しても、旧campaignの確認を新経路の採用へ自動変換しない。

## 検証

合成動画・擬似CLIで検証し、モデルAPI・実データ・GPUへアクセスしない。

```bash
.venv/bin/python -m pytest tests/unit/tennis_scene/chat_annotation/local_agent tests/e2e/tennis_scene/test_local_agent.py -n 4 -q
```
