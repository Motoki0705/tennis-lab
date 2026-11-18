#2253 の5モデル×3クリップ×4カメラの比較が完了しました。#1752 の保存済みの校正・コート・人物を読み込み、`court_side=[false,false,true,true]` を固定してボールを再構成しました。

可視化用の15個のNPZ、`manifest.json`、`summary.csv` は以下に保存しました。

```text
[local path]
```

| モデル | 3D有効（全883フレーム） | 指定範囲外 |
|---|---:|---:|
| 現行 `pipeline_x3d_dpt` | 410（46.4%） | 29 |
| 旧既定 `i935-convnext` | 426（48.2%） | 0 |
| `convnext-dpt-t64` | 350（39.6%） | 0 |
| `convnext-query-t64` | 595（67.4%） | 13 |
| `x3d-query-d512-l8` | 609（69.0%） | 111 |

範囲外は有効3D点のうち `|x| > 9 m` または `|y| > 18 m` の点です。対象動画に対応する2D正解は見つからなかったため、表は可用性と幾何の比較です。直接座標モデルの `raw_score` はNaNとし、軌道ゲートには各提案を重み1で渡しました。

現行モデルのgate後2D観測、平滑化、3D座標・マスク・棄却理由は #1752 と完全一致しました。生座標は保存ヒートマップのCPU再デコードと最大0.000122 px差で、同じ格子セルの保存GPU候補がある3,197点では完全一致しました。

ブランチは `codex/ball-model-compare`、HEADは `36699fc98` です。CPUテスト59件と対象HEADの `Python`・`Knowledge library` が成功し、PRを **Ready for review** にしました。[PR #2258](https://github.com/Motoki0705/tennis-lab/pull/2258) · [knowledge 000066](https://github.com/Motoki0705/tennis-lab/blob/36699fc98ddf4d0017915b0b6b6f725dc83600dc/knowledge/nodes/tennis_scene/000066-run-i2253-ball-model-compare-20261118.md)