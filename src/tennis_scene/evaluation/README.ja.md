# 固定入力によるball 3D評価

[English](README.md) · [简体中文](README.zh-CN.md)

`ball3d/` は同じ履歴componentから各方式をCPU replayし、人間提供の2D参照で採点する。
標準pipelineの既定を変更せず、検出器の推論を実行しない。

| モジュール | 契約 |
|---|---|
| `bundle.py` | 固定入力、元identity・source hash、完全な依存descriptorを抽出・検証 |
| `contracts.py` | 選択viewだけの方式入力とmask付き3D結果 |
| `baseline.py` | 保存済み検出観測からconsensus・LOESS・三角測量をreplayし、native mapの復号を別途照合 |
| `runner.py` | 全viewと3通りの2view LOVO、型付きchecksum保存読込 |
| `annotations.py` | 共通readerの選択結果を採点専用配列へ変換 |
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

`annotations.py` は厳格な [user-reference reader](../ground_truth/README.md) の薄い変換層。
`load_annotations(path, bundle, expected_sha256=...)` は検証済みbundleと動画hashから
`ReferenceIdentity` を構築し、`load_user_reference` と `UserReference.select` で
camera/frame軸を明示的に選ぶ。独自JSON schemaや出自検証を二重管理しない。
参照JSONとともに元CSVを保存しておく。

採点可否は `valid_for_scoring` のみで決める。人間承認の `estimated` も採点し、
`observed` へ偽装しない。除外・欠測座標はNaNのまま補間しない。
全fold・全区間で `all`・`observed`・`estimated` の分布、対象数・採点数・coverageを出す。
camera別イベントの元行・注記・推定区間を採点reportへ保持する。

既定区間は調整 `[0,505)`、主評価 `[505,1010)`、参考の全区間。
LOVOは除外cameraのみ採点し、foldの中央値を平均せず個別残差をpoolする。
中央値/p90、厳密に20pxを超える率、全view誤差、有効3D率、粗大再投影frame率、
隣接移動量・加速度とその外れ率、イベント付近の誤差、全runのCPU時間を出す。
粗大frame率は、採点できたframeのうち1つでも20pxを超えたframeの割合であり、
独立した3D正解に対する粗大誤差率ではない。運動のstencilは欠測や区間境界を跨がない。
診断の既定閾値65m/s、500m/s²、イベント半径3frameは保存し、後半では調整していない。

イベントの採点窓はkindごとの元camera行の窓の和集合とする。同じ事象を1frameずらして
記録した行があっても、1残差は1回だけ数える。`source_row_count` を物理イベント数と
解釈せず、`physical_event_count` は未確定としてnullを出す。元の注記は削除しない。

`annotation_oracle` は採点専用で、foldの保持cameraの採点可能な参照（推定を含む）を
同じ固定cameraで三角測量する。予測方式は呼ばない。近似校正と参照の誤差があるため、
数学的に保証された下限ではなく幾何参照である。

可視化はN方式を扱う。直接観測は緑の十字、承認済み推定は白い菱形、方式の再投影は色付き円。
cropはLOVO誤差で選び、除外cameraへの予測を色付き斜め十字でも表示する。軌跡は欠測を跨がない。
HTMLへ動画・図・表・指標を埋め込み、単独で閲覧できる。隣のファイルは任意のdownload用。
共通の動画readerとframe選択を再利用する。`MultiviewSceneRenderer` はSMPL/SceneResultを
要求するため、ball専用の比較では人物meshを生成せず俯瞰・側面パネルを構成する。

## 再現とテスト

clip固有の入口、テスト設計と実証は
[`knowledge/runs/run-ball3d-evaluation-foundation-20261103`](../../../knowledge/runs/run-ball3d-evaluation-foundation-20261103/)
に置く。repoを `PYTHONPATH` に入れて `run.py --help` を実行する。
`bundle` / `compare` は明示的なローカル入力と新規出力先を要求する。
compareには信頼済み記録の `--bundle-sha256`・`--annotations`・`--annotations-sha256` と履歴重み照合用の `--checkpoints` を指定する。実資産・全動画decode・replay検証は軽量CI fixtureと分ける。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/evaluation/ball3d \
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py \
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py \
  tests/integration/tennis_scene/test_ball3d_bundle.py \
  tests/integration/tennis_scene/test_ball3d_media.py
```
