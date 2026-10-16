# Scene reveal animation

学習済みNHTシーンを、撮影カメラ → SfM点群 → ガウシアン点群 → 写実レンダの順に「世界ができていく」動画として描画する。続けて、採用されたコートのアライメントを重ねてデータセットのラベル座標系を示し、最後にカメラを動かして任意視点で描画できる3Dシーンであることを示す。写実レンダまではコート座標で指定した俯瞰視点に固定する。

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.visualize_scene_reveal
# 視点調整は stills モードで数枚だけ描画する
.venv/bin/python -m src.synthetic_data_generation.scripts.visualize_scene_reveal \
  reveal.mode=stills 'reveal.still_times_seconds=[2,10,16,20,27,32,40]'
```

設定は [visualize_scene_reveal.yaml](../../configs/visualize_scene_reveal.yaml)。出力は `output_video`（mp4）と同名の `.json` メタデータ、stills モードでは拡張子を除いたディレクトリにPNGを書く。

## 段階

| phase | 見え方 | 描画 |
|---|---|---|
| `cameras` | 撮影軌跡と視錐台（`camera_frustum_stride` 台ごと）が撮影順に伸びる | tennis-lab |
| `sfm` | SfM点群（export の `points_scene.npy`）が注視点から放射状に現れる | tennis-lab |
| `sfm_to_gaussians` | カメラとSfM点群が消え、単色のガウシアン点群が現れる | 合成 |
| `colorize` | ガウシアン点の色が単色から学習済みの色へ | NHT |
| `splat` | 大きいガウシアンから順に、点から学習済みの形・不透明度へ育つ（粗→細） | NHT |
| `view_dependence` | shaderの視線入力を光軸方向の固定値から実際の視線へ | NHT |
| `hold` | 最終レンダ（`nht-render` の出力そのもの） | NHT |
| `alignment` | 最終レンダの上に、各コートの地面ライン（CourtKP20の地面14点の骨格）が伸び、続いてコート座標軸（x赤・y緑・z青）とコートIDが現れる | tennis-lab |
| `camera_move` | ラインを重ねたまま、視点から `camera_path.keyframes` を順に通ってカメラが動く | NHT + tennis-lab |

`splat` では各ガウシアンの不透明度も自分の成長に合わせて学習値へ移る。不透明度を全体で後から下げる段階を分けると、大きいガウシアン（背景やfloater）が不透明なまま画面を覆い、途中の絵が成立しないため（B00で確認）。

## 構成

| ファイル | 役割 |
|---|---|
| `contracts.py` | 要求・結果の型と検証 |
| `timeline.py` | 段階の長さ → フレームごとの進捗値（純関数） |
| `viewpoint.py` | コート座標の look-at → NHTシーン座標の OpenCV カメラ、カメラ移動の経路 |
| `sources.py` | 公開export と `alignment.json` の読み込み、SfM点群の切り出し |
| `layers.py` | SfM点（z-buffer）とカメラ線の手続き的レイヤー |
| `court_overlay.py` | 採用コートのライン・座標軸・ラベルのレイヤー |
| `compositor.py` | NHTのRGB/alphaと各レイヤーの合成 |
| `renderer.py` | NHT状態の重複排除・分割描画、動画/PNG出力 |
| `configuration.py` | Hydra境界 |

## 設計上の決まり

- **NHTの内部は読まない。** ガウシアンの操作は `nht-render` の公開リクエストに追加した `appearance_policy` / カメラごとの `appearance` で行う（契約は [NHT README](../../../../third_party/nht/README.md#standard-renderer)）。全進捗が1のとき通常の `nht-render` と同一の出力になることをB00で確認した（最大差0）。
- **同じNHT状態は1回だけ描画する。** 状態は外観の進捗とカメラ経路上の位置で決まる。`sfm_to_gaussians`〜`colorize` は1状態、`hold` と `alignment` も同じ1状態で済み、`camera_move` はフレームごとに1状態になる。`states_per_invocation` 状態ごとに1プロセスで描画し、使い終わった配列は `work_directory` から消す。
- **`alignment.json` だけを読む。** 必要なのはコート変換と metric/NHT の相似変換で、line evidence は使わない。そのため alignment 所有ディレクトリ全体の検証（`load_accepted_layout`）は行わない。採用されていない候補がlayoutに含まれていれば拒否する。
- 視点はアンカーコートの座標系（m、ネット中央が原点、+x がコート幅方向、+y が長さ方向、+z が上）で指定する。B00 では court-001 が court-000 の x≈-14.9 にあり、既定値は court-000 の外側の角から2面を見下ろす。
- **コートは学習データのラベルと同じ幾何で描く。** 各コートの `scene_from_court` と metric/NHT の相似変換で CourtKP20 の地面骨格を置く。ネットはアライメントで合わせていない（ポスト位置は会場で変わる）ので描かない。
- **ラインに奥行き判定をしない。** NHTのdepthはalphaで重み付けした期待値で、霞があるとカメラ側に寄る（B00ではライン上で5〜16%手前）。遮蔽と半透明を区別できないので、ラインは常に手前に描く。
- **カメラ移動は一定速度。** 視点とkeyframeを通る一様Catmull-Romスプライン（目と注視点を同時に補間）を、目と注視点の移動距離の和で弧長パラメータ化し、進捗をsmoothstepで緩める。keyframe間の距離が違っても速度が跳ばない。up と画角は視点のものを使う。
- **撮影範囲の外へ低く出ない。** B00 の撮影カメラは court-000 座標で x -23〜8 m、y -17.6〜16.4 m にあり、そのすぐ外にフェンスがある。低い keyframe をフェンスの外に置くと、目の前のフェンスが画面を覆う（B00で確認）。
