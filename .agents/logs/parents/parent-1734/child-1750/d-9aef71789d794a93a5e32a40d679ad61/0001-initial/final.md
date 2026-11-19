#1750の実装・受入検証を完了しました。PR: [#2153](https://github.com/Motoki0705/tennis-lab/pull/2153)

branch: `codex/dino-batch-api`[local path]`7ec8be454`

APIは `predict_batch(requests, *, batch_size=1) -> list[PersonDetectionResult]`。前処理後の形状が同じ連続入力をまとめ、異なる形状は分割します。paddingせず、入力順と画像ごとの座標復元を保ちます。

Meijiの4camera・8実画像で比較しました。事前許容差はscore `1e-4`、box `0.1` 元画像px。閾値採否・row順・重複統合の採否は厳密一致です。

| batch | score最大差 | box最大差 px | 8画像中央値 | peak VRAM allocated / reserved |
|---|---:|---:|---:|---:|
| 1 | 0 | 0 | 2.677秒 | 1.738 / 2.125 GiB |
| 2 | 1.79e-6 | 0.000244 | 2.544秒 | 2.587 / 3.238 GiB |
| 4 | 1.19e-6 | 0.000244 | 2.494秒 | 4.293 / 5.518 GiB |

CPU関連62件、pre-commit、knowledge検証、対象HEADのCI全14ジョブ（`Python`含む）が成功しました。

DINO submoduleは固定版で初期化し、既存CUDA拡張を再利用できました。実画像の重複dropは0件で、dropありはCPUテストで確認しています。

残作業は親による統合レビュー・merge・merge後main CIです。独立validatorは未指定のため試行0／完了0です。