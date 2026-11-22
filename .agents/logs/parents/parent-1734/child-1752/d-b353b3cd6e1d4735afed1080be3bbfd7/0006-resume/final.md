[PR #2227](https://github.com/Motoki0705/tennis-lab/pull/2227) は **Ready・競合なし**です。

- branch: `codex/video-batch-cli`
- HEAD: `18b034b951b9d92bd97b578abcc3c6e85e5f7fa1`
- mainのC9デコーダー照合と、C4のlock・export世代・配列検査を両立しました。
- 本番コマンドをplayer証拠・ball両機能off・両fp16へ更新。共有コートload、sideはclip単位です。
- CPU **1,826件成功[local path]、pre-commit成功。保存済み3clipの判定は完全一致。69clipは1計画・2,622 nodeの構築まで確認しました。
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44483100113) はPythonを含む全ジョブ成功。[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44483100097)も全成功です。

GPU・queue登録・69clip本番は未実行です。69clipのplayer判定、最新GPU検証、既報のBall補助差は未確認・未解決としてPRに残しています。validatorは指定／試行／完了とも0回です。