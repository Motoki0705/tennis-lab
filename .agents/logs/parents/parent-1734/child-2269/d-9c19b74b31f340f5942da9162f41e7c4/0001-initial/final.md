#2269を実装し、PR #2270を **Ready** にしました。ブランチは `codex/court-side-player`、HEADは `6d0d2d2e17d5c9015cc244cd6c86902b0a67b2b1` です。[PR #2270](https://github.com/Motoki0705/tennis-lab/pull/2270)

#1757で以下を指定すると、選手の足元だけでcourt sideを判定し、ボール検出・点生成・三角測量を計画から外せます。初回HEADを固定している場合は `court_side.player.match_distance_m=3.0` も指定すると、最終版と同じ閾値になります。

```bash
court_side.evidence=player \
ball_detection.enabled=false \
ball_reconstruction.enabled=false
```

meiji_4camでは、cam0〜cam3の順で**単独3clip・動画共有ともに `[F,F,T,T]`** でした。停止は0件です。表の各セルは `cost / support` です。

| 仮説 | clip_000 | clip_027 | clip_037 | 3clip共有 |
|---|---:|---:|---:|---:|
| FFFF | 0.4670 / 0 | 0.4212 / 0 | 0.5654 / 0 | 0.4852 / 0 |
| FFFT | 0.4094 / 0.0139 | 0.3747 / 0 | 0.4513 / 0 | 0.4127 / 0.0054 |
| FFTF | 0.3590 / 0.0625 | 0.3447 / 0 | 0.4465 / 0 | 0.3828 / 0.0244 |
| **FFTT** | **0.1856 / 0.2292** | **0.1075 / 0.8598** | **0.1314 / 0.3983** | **0.1456 / 0.4661** |
| FTFF | 0.4267 / 0 | 0.3318 / 0.1776 | 0.4762 / 0 | 0.4150 / 0.0515 |
| FTFT | 0.4743 / 0 | 0.4328 / 0 | 0.5790 / 0 | 0.4957 / 0 |
| FTTF | 0.4751 / 0 | 0.4344 / 0 | 0.5604 / 0 | 0.4906 / 0 |
| FTTT | 0.4068 / 0 | 0.3448 / 0 | 0.4623 / 0 | 0.4066 / 0 |
| **margin** | **0.1734** | **0.2242** | **0.3151** | **0.2372** |

初期の2.5 mでは、clip_037がsupport 0.161で下限0.2に届かず停止しました。試した2.5・3.0・3.5 mのうち、全3clipが確定した最小値の3.0 mを採用しました。共有検証には元の校正数値と一致する3clip用成果物を使用し、再実行でもスコアとdigestが一致しました。全69clipへの適用と#1757のGPU実測は未確認です。[選定根拠と再現手順](https://github.com/Motoki0705/tennis-lab/blob/6d0d2d2e17d5c9015cc244cd6c86902b0a67b2b1/knowledge/nodes/court_side/000008-run-i2269-player-court-side-20261119.md)

- 基準版のボール経路は全38段を同じ参照で再ロードでき、side出力と動画共有成果物のdigestも一致しました。
- ローカルの関連テスト1,463件、README検証33件、#1757との一括処理CPU検証22件、pre-commitが通過しました。
- **最終HEADの `Python` とKnowledge libraryは成功**しました。初回CIは後続pushでキャンセルされました。[最終CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44459366439)・[Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44459366416)