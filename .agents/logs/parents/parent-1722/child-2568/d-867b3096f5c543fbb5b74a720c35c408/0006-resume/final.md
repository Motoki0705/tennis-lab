レビュー修正後の `b682280df117b3018c50885a422642de03b72687` で、multiview-v2の **69クリップすべてを再実行・照合**しました。失敗は0件です。候補数は打球425件（差0）、バウンド363件（差0）、ネット接触34件（差−1）でした。`clip_015` のframe 802のネット候補が消え、別の6件は時刻が1〜2フレーム移動しました。 [クリップ別予測JSON]([local path]) · [処理状況]([local path])

#2542の固定ラベルと既存Refiner予測を使い、採点6本を±5フレームで再照合しました。規則のF1は打球0.868、バウンド0.714、ネット接触0.750で、採点指標JSONは初回とバイト一致しました。 [再採点結果]([local path])

| 欠測・雑音付き合成テスト | 適合率 | 再現率 | F1 | 時刻MAE |
|---|---:|---:|---:|---:|
| 打球 | 1.000 | 0.717 | 0.835 | 8.16 ms |
| バウンド | 0.998 | 0.876 | 0.933 | 10.26 ms |
| ネット接触 | 1.000 | 0.750 | 0.857 | 6.60 ms |

レビューで再現されたボレーは `shot` 1件・`net` 0件になりました。白帯に触れて越える球と落ちる球、FP32・ECPose・FastHMRのキャッシュ照合を含む23件とpre-commitが通りました。

- `codex/parent-1722-issue-2568-primary` の実装に、全件再実行と再採点の証拠を追記しました。 [PR #2569](https://github.com/Motoki0705/tennis-lab/pull/2569)
- `codex/parent-1722-issue-2568-knowledge` の `fecc5377c36ba2f4e7044ca1bd75a1c20dd69ddd` で、合成評価の再現と指標JSONのバイト一致を確認しました。 [PR #2570](https://github.com/Motoki0705/tennis-lab/pull/2570)

両PRのmainとの競合を解消し、新HEADのCIを起動しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2569,"head":"b682280df117b3018c50885a422642de03b72687","resume_on_success":false}}