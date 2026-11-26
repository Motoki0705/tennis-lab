[PR #2498](https://github.com/Motoki0705/tennis-lab/pull/2498) に生成結果と品質監査を追記しました。ブランチは `codex/parent-2492-issue-2495-primary`、HEAD は `b1011e406f97956425fe29b8b6ba987300459e96` です。

[読み取り専用の公開先]([local path]) に、代表 6 クリップの実推論 1,711 フレーム、4 カメラの 6,844 画像分を保存しました。`sha256sum -c SHA256SUMS` は全 66 件で一致しました。

| Manifest ID | SHA-256 |
|---|---|
| `meiji_4cam/multiview-player-evaluation-v1` | `3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580` |
| `meiji_4cam/multiview-player-evaluation-v1/body-recovery-inputs` | `0b823674c269b395f2e67de2cb84851e8dcba7409b928ced92b69639480c939f` |

- BBox は採点対象 26,276 件、除外 1,100 件（GVHMR 欠測 908、SMPL 無効 80、画像内関節不足 112）。関節は採点対象 437,714 点、除外 27,678 点でした。 [クリップ・カメラ・選手別の品質集計](https://github.com/Motoki0705/tennis-lab/blob/b1011e406f97956425fe29b8b6ba987300459e96/knowledge/runs/run-meiji-multiview-player-evaluation/quality_counts.csv)
- `clip_003` の frame 120 で座標合成の独立計算との差は最大 `4.60e-7 m`、OpenCV との射影差は `1e-6 px` 未満でした。全件の ViTPose smoke は Recall@0.50/0.75 が `0.9757/0.7770`、未検出込みの平均 OKS が `0.7707`。再採点は保存済みの全集計と一致しました。身体配置と品質ゲートが ViTPose 観測に依存するため、値は疑似 GT との一致度です。 [knowledge 実行記録](https://github.com/Motoki0705/tennis-lab/blob/b1011e406f97956425fe29b8b6ba987300459e96/knowledge/nodes/tennis_scene/000084-run-meiji-multiview-player-evaluation.md)
- 動画 24 本は公開先の `clips/<clip>[local path]` から `cam3.mp4` にあります。代表 PNG 2 枚で重ね描きとフレーム表示を確認しました。人間による全体の目視確認は親へ引き継ぎます。 [全動画と代表 PNG のパス](https://github.com/Motoki0705/tennis-lab/blob/b1011e406f97956425fe29b8b6ba987300459e96/knowledge/runs/run-meiji-multiview-player-evaluation/publication.json)

初回 CI の import エラーは既存の関数名を復元して修正しました。影響範囲の pytest 90 件、pre-commit、knowledge 検査は通過済みです。 [更新後の CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45383712107)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2498,"head":"b1011e406f97956425fe29b8b6ba987300459e96","resume_on_success":false}}