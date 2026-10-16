# Scene reveal animation

学習済みNHTシーンを、撮影カメラ → SfM点群 → ガウシアン点群 → 写実レンダの順に「世界ができていく」動画として描画する。カメラは固定で、コート座標で指定した俯瞰視点から見る。

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.visualize_scene_reveal
# 視点調整は stills モードで数枚だけ描画する
.venv/bin/python -m src.synthetic_data_generation.scripts.visualize_scene_reveal \
  reveal.mode=stills 'reveal.still_times_seconds=[2,10,16,20,27]'
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

`splat` では各ガウシアンの不透明度も自分の成長に合わせて学習値へ移る。不透明度を全体で後から下げる段階を分けると、大きいガウシアン（背景やfloater）が不透明なまま画面を覆い、途中の絵が成立しないため（B00で確認）。

## 構成

| ファイル | 役割 |
|---|---|
| `contracts.py` | 要求・結果の型と検証 |
| `timeline.py` | 段階の長さ → フレームごとの進捗値（純関数） |
| `viewpoint.py` | コート座標の look-at → NHTシーン座標の OpenCV カメラ |
| `sources.py` | 公開export と `alignment.json` の読み込み、SfM点群の切り出し |
| `layers.py` | SfM点（z-buffer）とカメラ線の手続き的レイヤー |
| `compositor.py` | NHTのRGB/alphaと各レイヤーの合成 |
| `renderer.py` | NHT状態の重複排除・分割描画、動画/PNG出力 |
| `configuration.py` | Hydra境界 |

## 設計上の決まり

- **NHTの内部は読まない。** ガウシアンの操作は `nht-render` の公開リクエストに追加した `appearance_policy` / カメラごとの `appearance` で行う（契約は [NHT README](../../../../third_party/nht/README.md#standard-renderer)）。全進捗が1のとき通常の `nht-render` と同一の出力になることをB00で確認した（最大差0）。
- **同じNHT状態は1回だけ描画する。** カメラ固定なので、`sfm_to_gaussians`〜`colorize` と `hold` は1状態で済む。`states_per_invocation` 状態ごとに1プロセスで描画し、使い終わった配列は `work_directory` から消す。
- **`alignment.json` だけを読む。** 必要なのはコート変換と metric/NHT の相似変換で、line evidence は使わない。そのため alignment 所有ディレクトリ全体の検証（`load_accepted_layout`）は行わない。採用されていない候補がlayoutに含まれていれば拒否する。
- 視点はアンカーコートの座標系（m、ネット中央が原点、+x がコート幅方向、+y が長さ方向、+z が上）で指定する。B00 では court-001 が court-000 の x≈-14.9 にあり、既定値は court-000 の外側の角から2面を見下ろす。
