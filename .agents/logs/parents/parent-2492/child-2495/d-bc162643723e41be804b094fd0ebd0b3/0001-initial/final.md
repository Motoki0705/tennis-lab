[ドラフト PR #2498](https://github.com/Motoki0705/tennis-lab/pull/2498) に、SMPL 由来の疑似 GT、FastHMR 固定入力、検出・COCO17 採点器を追加しました。branch は `codex/parent-2492-issue-2495-primary`、HEAD は `dfbacf508967003eaa3cecb98c48dab44eeb7e42` です。

代表 6 クリップの保存済み GVHMR 出力から、実推論 1,711 フレーム・6,844 画像を確認しました。`clip_028` の 4 人同時有効率は元データの実測で 99.46%（1,281/1,288）でした。疑似 BBox は各辺 10% の余白と画像内 12 関節で固定し、遮蔽は手前の選手の胴体と観測欠損から推定します。 [生成手順と品質規則](https://github.com/Motoki0705/tennis-lab/blob/dfbacf508967003eaa3cecb98c48dab44eeb7e42/knowledge/runs/run-meiji-multiview-player-evaluation/README.md)

pytest 22 件と pre-commit が通過しました。`clip_003` の frame 120 では、独立計算との座標差は最大 `4.60e-7 m`、OpenCV との射影差は全カメラで `1e-6 px` 未満でした。同フレームの ViTPose smoke は Recall@0.50 が 0.9375、未検出を含む平均 OKS が 0.7348 でした。

全件生成ジョブを起動しました。公開先は `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-player-evaluation-v1` です。ジョブ完了後に manifest の SHA-256、24 本の動画、全件の品質統計と smoke を照合し、knowledge の実行記録と PR に追記します。CI も結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T110738Z_96f7ca821d94_multiview-player-evaluation"]}}