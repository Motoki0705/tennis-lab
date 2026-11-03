# 固定入力によるball 3D評価

[English](README.md) · [简体中文](README.zh-CN.md)

`ball3d/` は同じ履歴componentから各方式をCPU replayし、人間提供の2D参照で採点する。
標準pipelineの既定を変更せず、検出器の推論を実行しない。

| モジュール | 契約 |
|---|---|
| `bundle.py` | 固定入力、元identity・source hash、完全な依存descriptorを抽出・検証 |
| `contracts.py` | 選択viewだけの方式入力とmask付き3D結果 |
| `baseline.py` | 保存済み検出観測からconsensus・LOESS・三角測量をreplayし、native mapの復号を別途照合 |
| `court_ball_ba.py` | 任意のproduction [court/ball bundle adjustment](../ball_calibration/README.ja.md) へのadapter |
| `runner.py` | 全viewと3通りの2view LOVO、型付きchecksum保存読込 |
| `annotations.py` | 共通readerの選択結果を採点専用配列へ変換 |
| `scoring.py` | 区間別の再投影・coverage・運動・イベント診断、注釈のみの幾何参照 |
| [比較可視化](../visualization/ball3d/) | N方式のRGB重畳、俯瞰・側面軌道、図・crop・持ち運べるHTML |

`segment_physics.py` は任意の [区間物理軌道・相対時刻solver](../ball_trajectory/README.ja.md) を選択viewだけの初期値で実行し、共通の結果形式へ渡します。

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

任意の [`TopKMultiHypothesis`](ball3d/topk.py) adapterはproductionの
[top-K対応付け](../ball_association/README.ja.md)を呼び、loaderや採点を追加しません。

任意選択の[候補GMM方式](../ball_gmm/README.ja.md)はproductionの方式Aを使います。
`ball3d/uncertainty.py` は同じ採点camera/frame集合で投影共分散と楕円被覆率を評価します。
非ゼロ時刻offsetには時間方向の同時事後分布が必要なため、明示拒否します。

`BallMethod.name`、`config`、boolの `requires_court_lines`、`__call__(inputs) -> Ball3DResult` を実装し、bundle/codeの
digestとともに `run_method` / `run_comparison` へ渡す。runnerに注釈引数はない。
LOVOでは保持する2つの `ViewInput` だけを渡し、除外cameraのheatmapや観測は渡さない。
元runの固定校正は全方式で共有する。

各viewはaligned camera、local校正/side、モデルevidence、閾値/gate後の保存済み検出点、
採用ball点と記録されたreplay設定を持つ。座標は元動画の画素。heatmapは固有のH/W格子を保ち、
候補の画素変換はdetectorのW−1/H−1規約を用いる。全source frameと読み取り専用配列を保持する。

`ViewInput.court` はcamera別のraw KP14画素・score/valid・元homographyの採用subsetと
テンプレートの向きを、実際の観測frame（今回は0）だけで保持する。fitted/broadcast座標をrawと偽らない。
camera/frame/nativeサイズ/欠落を検証し、LOVOは選択viewのcourtだけを渡す。
raw KPは既存bundleの検証済み校正diagnosticsに残っており、bundleのbytes/digestは変更しない。
LINEは型付きで保持できるが、元storeにはsupport要約しかない。`court.lines=None` と欠落理由を保持する。
各方式はboolの `requires_court_lines` を必ず宣言し、runnerはtrueなら選択viewの欠落時に方式を呼ぶ前に停止する。
直接の要求は `court.require_lines()` または `load_bundle(..., require_court_lines=True)` で行える。
baselineはfalseで固定cameraを使う。要約をLINE観測に変換したり推論を再実行しない。

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

`score_comparison(runs, bundle, annotations)` と `error_samples(run, bundle, annotations)` は
検証済み `FixedInputBundle` を必須とし、runのbundle digest・camera/frame軸と、正解の
clip ID・camera・frame数・fps・nativeサイズ・動画hashを照合する。`annotation_oracle`・
`plot_comparison`・`render_extremes` もbundleを取る。同サイズの別入力の混入を拒否する。

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

`paired_error_summary(reference, candidate, bundle, annotations, ...)` も `error_samples` と
同じbundle/結果/正解identity照合を行う。2組の共通集合は、全方式の共通LOVO cohortとは区別する。
2組のLOVOのcamera/frame残差の共通集合を、全foldまたは
除外camera別に比べる。各組の採点数と参照−比較の差を残す。coverageが異なる場合の
非加算的な診断であり、clip固有の誤差分解は
[`knowledge/runs/run-ball3d-error-decomposition-20261103`](../../../knowledge/runs/run-ball3d-error-decomposition-20261103/) に置く。

`annotation_oracle` は採点専用で、foldの保持cameraの採点可能な参照（推定を含む）を
同じ固定cameraで三角測量する。予測方式は呼ばない。近似校正と参照の誤差があるため、
数学的に保証された下限ではなく幾何参照である。

HTML主表と単独の指標図は共通の表データを使い、LOVO誤差の隣に採点数・対象数・coverageを出す。
全view有効3D率は別の見出しにする。第2表は全登録方式（含まれる場合はoracleも）の採点可能な
(除外camera, 元frame)集合の共通部分だけで比較する。方式間の採点集合が異なること、共通集合でも
欠測を回復した比較にはならないことを表の近くに明記する。共通集合のidentityと観測/推定の内訳も保存し、
共通点が0なら誤差は未測定とする。

可視化はN方式を扱う。直接観測は緑の十字、承認済み推定は白い菱形、方式の再投影は色付き円。
cropはLOVO誤差で選び、除外cameraへの予測を色付き斜め十字でも表示する。軌跡は欠測を跨がない。
HTMLへ動画・図・表・指標を埋め込み、単独で閲覧できる。隣のファイルは任意のdownload用。
共通の動画readerとframe選択を再利用する。`MultiviewSceneRenderer` はSMPL/SceneResultを
要求するため、ball専用の比較では人物meshを生成せず俯瞰・側面パネルを構成する。

## 再現とテスト

clip固有の入口、テスト設計と実証は
[`knowledge/runs/run-ball3d-foundation-review-20261103`](../../../knowledge/runs/run-ball3d-foundation-review-20261103/)
に置く。repoを `PYTHONPATH` に入れて `run.py --help` を実行する。
`bundle` / `compare` は明示的なローカル入力と新規出力先を要求する。
compareには信頼済み記録の `--bundle-sha256`・`--annotations`・`--annotations-sha256` と履歴重み照合用の `--checkpoints` を指定する。修正後の再現では `--prior-comparison` でv1予測配列との一致も照合する。実資産・全動画decode・replay検証は軽量CI fixtureと分ける。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/evaluation/ball3d \
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py \
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py \
  tests/integration/tennis_scene/test_ball3d_bundle.py \
  tests/integration/tennis_scene/test_ball3d_media.py
```
