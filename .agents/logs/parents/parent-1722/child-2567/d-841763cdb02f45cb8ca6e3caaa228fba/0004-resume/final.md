#2567 の学習と `clip_000` の推論を完了しました。実装は [PR #2571](https://github.com/Motoki0705/tennis-lab/pull/2571)、実験記録は [PR #2574](https://github.com/Motoki0705/tennis-lab/pull/2574) です。学習に使った実装HEADは `e4438667dd41162c6f4972a92146eb9b59bbba54` です。

新データ版の1,280ラリーにはネット接触が640件あり、白帯を越えた球と打者側に落ちた球が320件ずつでした。GPU学習は4,000更新を688.2秒で完了しました。合成test 128ラリーの保存予測を再集計し、診断JSONとの一致を確認しました。

| クラス | 適合率 | 再現率 | F1 | 平均絶対時刻誤差 |
|---|---:|---:|---:|---:|
| ショット | 0.978 | 0.891 | 0.932 | 7.23 ms |
| バウンド | 0.974 | 0.998 | 0.986 | 3.73 ms |
| ネット接触 | 0.871 | 0.844 | 0.857 | 11.73 ms |

ネット接触は白帯を越えた32件中29件、打者側に落ちた32件中25件が一致しました。バウンドの採点は出力フレーム内の497件で、最終出力フレームより後の25件は物理記録に残っています。

**#2542への引き渡し**

```text
[local path]
```

SHA-256は `b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0` です。`RefinerPredictor.from_checkpoint()` で読み込み、`inference.scene.predict_scene_events()` に `scene.json`、`clip_id`、`checkpoint_sha256`、`seed=42`、`fps_tolerance=0.06` を渡します。

`clip_000` は欠測0の356フレームで、出力確率は `(1, 356, 3)` でした。59.94005994 fpsの元フレームを保ち、閾値0.5でショット2件、バウンド3件、ネット接触1件を検出しました。ネット接触候補はframe 327、確率0.729です。再実行でJSONとNPZがバイト一致しました。実映像の採点は #2542 で行います。 [イベントJSON]([local path]) [クラス別確率NPZ]([local path])

ナレッジ記録の構造検証と再現パス検査は通過し、記録PRのCI待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2574,"head":"2e01e36a032b62c696e2c8593a4bfec2bb8ce8b4","resume_on_success":false}}