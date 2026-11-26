RF-DETR-L の PT16・TRT16 × batch 1・2 を、DINOと同じ固定96画像・CPUスレッド設定で計測しました。4条件すべてで各300反復の入力順と集計を確認しました。

| Batch 1 | Wall P50（ms/画像） RF-DETR-L / DINO | NVMLピーク増分（MiB） RF-DETR-L / DINO | Recall@0.50 RF-DETR-L / DINO |
|---|---:|---:|---:|
| PT16 | 39.77 / 165.69 | 368 / 2,406 | 96.82% / 96.69% |
| TRT16 | 26.43 / 94.79 | 142 / 984 | 96.81% / 96.67% |

RF-DETR-Lのbatch 1は、DINOよりwall P50がPT16で76.0%、TRT16で72.1%短く、Recall@0.75は両条件で約5ポイント低い結果でした。検出score下限はRF-DETR-Lが0.25、DINOが0.30です。batch 2の精度と下流品質は未検証で、新規計測のparityは「PT32未計測」です。[全条件の比較表](https://github.com/Motoki0705/tennis-lab/blob/f92c40d600359e4920949ed70704a3f840a3cc3e/knowledge/runs/run-meiji-rfdetr-large-speed/comparison.csv)

`codex/parent-2492-issue-2487-rfdetr` に `f92c40d600359e4920949ed70704a3f840a3cc3e` をpushしました。batch 1の2レポートは #2499 の集計で受理されることを確認し、対象テスト62件とknowledge検証が通過しました。[実行証拠と成果物のSHA-256](https://github.com/Motoki0705/tennis-lab/blob/f92c40d600359e4920949ed70704a3f840a3cc3e/knowledge/runs/run-meiji-rfdetr-large-speed/results.json) · [起動済みCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45624206690)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}