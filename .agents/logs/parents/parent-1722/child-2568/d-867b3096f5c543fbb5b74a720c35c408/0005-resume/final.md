mainとの競合を解消し、固定ハッシュによるFP32互換テストと、ECPose・FastHMRのキャッシュ照合を両立させました。ネット直後のボレーは、速度変化が起きたフレームと接触後の移動距離で識別します。レビューの再現条件で `shot` 1件、`net` 0件になり、白帯に触れて越える球と落ちる球も検出しました。関連テスト23件とpre-commitが通りました。

固定の欠測・雑音付き合成テストを再集計しました。

| 種別 | 適合率 | 再現率 | F1 | 時刻MAE |
|---|---:|---:|---:|---:|
| 打球 | 1.000 | 0.717 | 0.835 | 8.16 ms |
| バウンド | 0.998 | 0.876 | 0.933 | 10.26 ms |
| ネット接触 | 1.000 | 0.750 | 0.857 | 6.60 ms |

実装commit `b682280df117b3018c50885a422642de03b72687` で69件の再実行を開始しました。出力を照合した後、#2542の既存ラベルとRefiner予測を使って再採点するジョブです。 [新しい予測の出力先]([local path]) · [再採点の出力先]([local path])

合成評価の記録commit `fecc5377c36ba2f4e7044ca1bd75a1c20dd69ddd` は再現実行で指標JSONのバイト一致を確認しました。ジョブ完了後に候補数の差分と再採点結果を確認し、既存の2本のPRを更新します。 [実装 PR #2569](https://github.com/Motoki0705/tennis-lab/pull/2569) · [記録 PR #2570](https://github.com/Motoki0705/tennis-lab/pull/2570)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T010440Z_8bfcf27dbb18_meiji-net-contact-review"]}}