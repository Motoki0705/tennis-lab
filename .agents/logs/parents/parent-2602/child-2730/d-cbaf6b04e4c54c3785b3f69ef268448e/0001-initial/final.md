#2730 の Pillar I・II 11ページを改訂し、公開前検証を完了しました。冒頭定型表は11→0件、定型戻り行は9→0件、本文の表は38→22件です。

| 改訂ファイル | 表数（冒頭込み、旧→新） |
|---|---:|
| `tmp/wiki/Tennis-Scene-Pipeline-Overview.md` | 5→1 |
| `tmp/wiki/Tennis-Scene-History-and-Optimizations.md` | 4→0 |
| `tmp/wiki/Component-Person-Detection-and-Tracking.md` | 3→1 |
| `tmp/wiki/Component-CourtSide-Association-Alignment.md` | 3→1 |
| `tmp/wiki/Component-3D-Reconstruction-and-Tooling.md` | 3→0 |
| `tmp/wiki/Court-Detection-Overview.md` | 4→0 |
| `tmp/wiki/Court-Model-Evolution-and-Backbones.md` | 7→7 |
| `tmp/wiki/Court-Geometry-Postprocessing-and-Video-Shared.md` | 6→2 |
| `tmp/wiki/Synthetic-3DGS-Alignment-and-Generation.md` | 6→5 |
| `tmp/wiki/Synthetic-Scaling-Diversity-and-Pair-Studies.md` | 5→4 |
| `tmp/wiki/NHT-Rendering-and-Foreground-Compositing.md` | 3→1 |

- `tools.wiki_assets verify` は **11/11 PASS**。旧版のリンク227件、見出しアンカー72個、数値・評価条件・採否判断を突合し、欠落は0件でした。Tier 1 公開後の Wiki `master` `99b66328` でも担当11ページの旧版が一致しました。 [監査記録](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/rollout-scene-court/audit.md) · [照合 JSON](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/rollout-scene-court/audit.json)
- ローカル Chromium のライト・ダーク両方で、横スクロールは本文幅896pxで **2→0件**、390pxで **26→0件**、追加測定の326pxで **34→0件**。390pxでの代表3ページの高さは、標準パイプライン **8,746→6,315px**、コートモデル比較 **7,138→7,327px**、全6ペア比較 **6,430→5,857px** でした。代表3ページの前後24枚を目視し、折りたたみ8件の開閉と Mermaid 8図の表示も確認しました。 [表示計測とスクリーンショット](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/rollout-scene-court/render-check.md)

全38ページの公開監査は統合工程に残っています。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}