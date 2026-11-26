DINO＋ViTPose-Hの一体計測4条件を検証し、36条件の暫定比較表と反復データを更新しました。各条件で固定96画像を使った300反復が完了し、採点時の重みとTensorRT planのSHA-256も一致しました。[PR #2499](https://github.com/Motoki0705/tennis-lab/pull/2499)

| Backend | 画像batch | P50 / P95（ms/画像） | 画像/秒 | NVMLピーク増分 |
|---|---:|---:|---:|---:|
| PT16 | 1 | 266.30 / 359.32 | 3.51 | 6,484 MiB |
| PT16 | 2 | 270.61 / 352.13 | 3.52 | 7,618 MiB |
| TRT16 | 1 | 148.58 / 194.14 | 6.41 | 2,292 MiB |
| TRT16 | 2 | 145.65 / 182.93 | 6.62 | 2,284 MiB |

4条件のparityは「PT32未計測」です。初回PT32の590.33 ms/画像は、同期境界が異なる旧harnessの暫定値として記録しています。候補結果の照合とPareto図の確定が残っています。

対象テスト44件、pre-commit、knowledge検証が通過しました。`codex/parent-2492-issue-2487-primary` に `00de353607ba62224e51400a05e6702f151869dc` をpush済みです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"00de353607ba62224e51400a05e6702f151869dc","resume_on_success":true,"next_step":"補助担当の確定した計測結果を照合し、比較表とPareto図に反映する。"}}