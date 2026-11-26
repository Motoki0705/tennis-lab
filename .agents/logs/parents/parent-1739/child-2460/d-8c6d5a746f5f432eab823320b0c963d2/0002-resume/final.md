#2460 はユーザー判断どおり、clip_000・clip_020 のGPU試行で止めました。両clipでGVHMR・身体配置・`scene_assembly` を読み戻しました。

| clip | 選手数 | 3D選手フレーム | 有効root | 有効メッシュ | GPU処理 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clip_000 | 4 | 1,416 | 1,410 | 1,407 | 14.07秒 |
| clip_020 | 4 | 2,579 | 2,575 | 2,550 | 114.35秒 |

新しいsceneは別ストアに保存しました。各clipのボール5配列と、共有dataset全69件の `scene.json`・`annotation.json` は #2458 版とバイト一致しました。CPU全件の結果、試行結果、全件投入の再開手順をknowledgeに記録し、[#2460 の停止事項](https://github.com/Motoki0705/tennis-lab/issues/2460)を更新しました。

pytest 788件、pre-commit、knowledge検査が通過しました。[draft PR #2471](https://github.com/Motoki0705/tennis-lab/pull/2471)のCI待ちです。

HEAD `61fef16091952b5c0df06f89b32603656a29a745`

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2471,"head":"61fef16091952b5c0df06f89b32603656a29a745","resume_on_success":false}}