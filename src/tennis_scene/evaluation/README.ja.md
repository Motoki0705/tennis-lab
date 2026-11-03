# 固定入力によるball 3D評価

[English](README.md) · [简体中文](README.zh-CN.md)

`ball3d/` は同じ履歴componentから各方式をCPU replayし、独立な2D観測で採点する。
標準pipelineの既定を変更せず、検出器の推論を実行しない。

| モジュール | 契約 |
|---|---|
| `bundle.py` | 固定入力、元identity・source hash、完全な依存descriptorを抽出・検証 |
| `contracts.py` | 選択viewだけの方式入力とmask付き3D結果 |
| `baseline.py` | 保存済み検出観測からconsensus・LOESS・三角測量をreplayし、native mapの復号を別途照合 |
| `runner.py` | 全viewと3通りの2view LOVO、型付きchecksum保存読込 |
| `annotations.py` | 採点専用のnative画素ラベル、確認状態と出自 |
| `scoring.py` | 区間別の再投影・coverage・運動・イベント診断、注釈のみの幾何参照 |
| [比較可視化](../visualization/ball3d/) | N方式のRGB重畳、俯瞰・側面軌道、図・crop・持ち運べるHTML |

## 固定入力と出自

`create_bundle` は元の `scene.json`、選択componentのdescriptorと配列、および推移的な
依存のdescriptorをコピーする。元のpath・identity・code出自・checksumを保持し、
過去storeを現在のpipeline実行と偽装しない。元storeは変更しない。
選択されない依存の配列はreplayに不要なのでコピーしない。

loaderはsource文書、依存鎖、ファイル一覧、配列checksum/型、camera/frame軸、
evidenceの窓の出自、side・校正・観測maskを検証する。注釈import由来や未宣言ファイルは拒否する。
受け渡し時は外部に記録したdigestを `load_bundle(path, expected_sha256=...)` へ渡す。
動画の移設は `verify_videos` へcamera ID別のpathを明示する。履歴の動画pathを自動探索しない。

`FixedInputBundle` は評価側の容器で、`baseline_reference` はreplay照合専用の保存済み予測。
方式へ渡すのは `inputs` だけ。参照3D・注釈・注釈の欠測状態・store pathは
`MethodInput` に存在しない。

## 方式インターフェース

`BallMethod.name`、`config`、`__call__(inputs) -> Ball3DResult` を実装し、bundle/codeの
digestとともに `run_method` / `run_comparison` へ渡す。runnerに注釈引数はない。
LOVOでは保持する2つの `ViewInput` だけを渡し、除外cameraのheatmapや観測は渡さない。
元runの固定校正は全方式で共有する。

各viewはaligned camera、local校正/side、モデルevidence、閾値/gate後の保存済み検出点、
採用ball点と記録されたreplay設定を持つ。座標は元動画の画素。heatmapは固有のH/W格子を保ち、
候補の画素変換はdetectorのW−1/H−1規約を用いる。全source frameと読み取り専用配列を保持する。

結果はコートmetreの `positions (T,3)`、`valid (T)`、明示的なcamera順の
`used_views (V,T)`。無効座標は0、有効点には2view以上が必要。
任意fieldはPSD共分散 `(T,3,3)`、変更後camera、cameraごとの秒単位offset、診断。
校正変更でも元のcourt座標規約を維持する。camera frame `t` は世界時刻 `t/fps + offset` を観測する。
小数offsetの投影では隣接する有効3D点だけを補間し、欠測越え・外挿はしない。
除外cameraは元校正・offset 0を保持する。

baselineは保存済み復号/gate後観測からproductionのconsensus・LOESS・三角測量を再計算する。
`verify_baseline` はnative mapの再復号、保存点・3D・maskを別途照合し、許容誤差とbit一致を区別する。
`without_loess` は同じconsensus採用maskを保ち、採用した元検出座標を使う。LOESSの座標変更だけを分離する。

## 採点ラベル

`load_annotations` は `tennis_lab_ball2d_annotations.v1` JSONを読む。

- `clip_id`、順序付き `camera_ids`、`num_frames`、`fps`、`source_size_wh`、camera別
  `video_sha256`、`kind: direct_video_review`・`model`・`effort` を含む `provenance`。
- `observations` は `camera_id`、`frame_index`、`status`、任意の `uv_px`、`reviewed`、`evidence`。
  状態は `observed` / `occluded` / `out_of_frame` / `unknown` / `unreviewed`。
  確認済み状態には証拠が必要で、座標を持つのはobservedだけ。未記載はunreviewed。
- 任意の `events` は `frame_index`、`kind`（`bounce` / `hit`）、`evidence`。

重複、source/軸の不一致、未確認の観測化、画像外座標、detector/test fixture由来を拒否する。
schemaは確認証拠を記録するが、画像を見たという主張の真実性を単独で証明しない。
unit fixtureを実評価ラベルに使わない。

既定は前半 `[0,505)` が調整、後半 `[505,1010)` が主評価、全区間は参考。
可視observedだけを採点し、LOVOでは除外viewだけを採点する。foldの中央値を平均せず、個々の残差をpoolする。
中央値/p90・20pxを厳密に超える率、observed/採点件数とcoverage、全view再投影、有効3D率、
frame粗大再投影率、隣接移動/加速度、イベント近傍と全run計算時間を出す。
運動のstencilは欠測・区間境界を跨がない。既定の診断閾値は65m/s、500m/s²、イベント半径3frameで、
記録に残し、評価後半で調整していない。

`annotation_oracle` は採点側に限り、各foldで保持viewの注釈だけを三角測量する。
近似校正・注釈誤差があるため、数学的に保証された誤差下限ではなく幾何参照として扱う。
注釈・イベントがなければ未計測とcoverageを返し、0で埋めない。独立3D GTなしに3D精度は測れない。

## 再現とテスト

clip固有の入口、テスト設計と実証は
[`knowledge/runs/run-ball3d-evaluation-foundation-20261103`](../../../knowledge/runs/run-ball3d-evaluation-foundation-20261103/)
に置く。repoを `PYTHONPATH` に入れて `run.py --help` を実行する。
`bundle` / `compare` は明示的なローカル入力と新規出力先を要求する。
独立な直接確認ラベルが届いてから `--annotations` を加える。実資産・全動画decode・replay検証は軽量CI fixtureと分ける。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/evaluation/ball3d \
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py \
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py \
  tests/integration/tennis_scene/test_ball3d_bundle.py \
  tests/integration/tennis_scene/test_ball3d_media.py
```
