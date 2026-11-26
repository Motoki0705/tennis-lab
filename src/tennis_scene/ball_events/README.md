# 規則によるボールイベント検出

`detect_scene_events(scene)` は [SceneResult v2・v3](../README.md#sceneresult-v3)
から打球（`shot`）、バウンド（`bounce`）、ネット接触（`net`）の候補を返す。
`detect_ball_events(ball_3d, valid, fps)` は合成軌跡などのボールのみの入力に使う。
どちらもコート座標のメートル単位で判定し、入力の0始まりのframeを保持する。

- バウンドは[既存の高さ極小検出](../../utils/rendering/trajectory_analysis.py)を使い、鉛直速度の負から正への反転を確認する。
- 打球は水平速度の方向または速さの急変を要求する。トス後の加速にも対応する。
  全選手の有効なCOCO17手首とrootまでの距離を調べ、使った部位・距離・クリップ内track IDを記録する。
  選手が遠い（`player_far`）か位置が欠測（`player_unavailable`）の候補には、ボールのみの閾値を適用する。
- ネット接触は[コート定義](../../utils/schema/court.py)のネット面・高さ・幅に従う。
  ネット高以下でのY速度の反射または急減、あるいは白帯近傍での減速と角度変化を求める。
  観測済みの候補では、直前・直後の1 frame間隔の速度差が長い速度窓での変化と同じ向きに
  十分現れていることも確認する。ネットを越えた側の候補は、接触後の1 frameで移動できる
  距離と測位誤差の許容幅から接触位置を絞る。
- 短い欠測は両側の速度から接触を推定し、`spans_ball_gap`、欠測を挟むframe、
  欠測数と外挿の食い違いを記録する。速度の計測区間は連続した実観測に限る。

既定値は [`RuleThresholds`](rules.py) にある。出力するscoreは閾値に対する判定特徴の強さを
0–1へ変換した値で、欠測を挟む候補は減点する。

## 固定ベンチマークで実行する

```bash
.venv/bin/python -m src.tennis_scene.ball_events \
  --benchmark data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v2 \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --output-root outputs --run-id meiji-multiview-v2-rule
```

指定したSHA-256にmanifestを照合し、各clipのscene index、アーカイブ、有効性、
fpsとframe数を検査する。`--clip-id video_000/clip_012` を指定すると対象を選べる。
複数指定する場合はオプションを繰り返す。

`<output-root>/tennis_scene/analyze/rule_ball_events/<run-id>/` 配下に
`predictions/<video_id>/<clip_id>.json` と `progress.json` を保存する。
予測は `ball_event_predictions.v1` で、`inputs.scene_sha256` は `scene.json` のSHA-256、
`method_version` は実行時checkoutのcommitを指す。再現用の実行はコミット済みのコードを使う。
clip単位の入力エラーは `status: failed` と理由を保存して続行する。失敗が残れば終了値は1になる。
進捗には全clip ID、処理状態、閾値、成功分の有効frame数とクラス別イベント数を記録する。
再実行するときは新しいrun IDを指定する。
