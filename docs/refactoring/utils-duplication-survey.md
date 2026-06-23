# 共通ユーティリティ重複調査

**日付:** 2026-06-23
**対象範囲:** `src/`（`src/tasks/`, `src/tennis_scene/`, `src/utils/`）
**きっかけ:** `src/tasks/court_detection/models/dinov3_detr.py:36` の `_resolve_project_path` — プロジェクトルートを基準にパスを解決する private ヘルパで、いかにも汎用的に見える。これが「他にもどれだけの汎用ロジックがタスクごとにローカル再実装されているのか？」という疑問のきっかけになった。

> 本書は**読み取り専用の棚卸し（census）と提案レポート**であり、ここではコードを移動していない。後続の抽出 PR を、クラスタ単位で一つずつ進めるための土台とすることを意図している。

---

## 要約（TL;DR）

本リポジトリにはすでに健全な共通パッケージ（`src/utils/`: `data/`, `models/`, `projection/`, `rendering/`, `schema/`, `video/`, `tensor_utils.py`）が存在する。しかし、**横断的な小さなヘルパ**の層が `src/utils/` に昇格されておらず、`ball_detection` / `court_detection` / `blcs` / `plcs` / `base` / `tennis_scene` にまたがってローカル再実装されている（場合によっては 1 バイトも違わない完全コピー）。

価値の高いギャップを、おおよそ優先度順に挙げる:

| # | クラスタ | 根拠 | 推奨配置先 |
|---|---------|------|-----------|
| 1 | **プロジェクトルート / パス解決**（今回のきっかけ） | `_resolve_project_path` + 約85箇所の `Path(__file__).parents[...]` / `to_absolute_path` | `src/utils/paths.py`（新規） |
| 2 | **デバイス解決** | 正典は `base/inference/predictor.py:93`、plcs スクリプトに**独立した3コピー**（戻り値が `str` か `torch.device` かで揺れ） | `src/utils/device.py`（新規） |
| 3 | **決定論的シード設定** | `_seed_everything` が**完全一致で2コピー**（blcs + plcs の `generate_dataset.py`）+ `base` runner のメソッド | `src/utils/seeding.py`（新規） |
| 4 | **`src/utils` 内部にすでにある重複** | `normalize_tensor_imagenet`（video）と `normalize_tensor_images_imagenet`（data）が**完全一致** | 1つに集約 |
| 5 | **`_clone_sample`** | `blcs` と `plcs` の augmentation で**完全一致** | `src/utils/tensor_utils.py` |
| 6 | **回転 / 角度の幾何計算** | `angular_error`、ラップ角差分、回転行列、骨格角が plcs + tennis_scene + 分析スクリプトに散在 | `src/utils/geometry/`（新規） |
| 7 | **JSON 結果保存の定型句** | `mkdir(parents=True) + json.dump(indent=2)` がパイプライン `Result.save()` の**6箇所**にコピペ | `src/utils/io.py`（新規） |
| 8 | **ヒートマップ→画素座標の逆正規化** | `heatmaps_to_argmax` + 手動 `*(W-1)` が4箇所以上で再実装 | `src/utils/data/heatmaps.py` を拡張 |
| 9 | **テンソル→numpy / ディレクトリ生成のイディオム** | `_to_numpy` ×3、`.detach().cpu().numpy()` のインライン、`mkdir(parents=True, exist_ok=True)` ×54 | `tensor_utils.py` / `io.py` |

---

## 調査方法

3体の Sonnet サブエージェントがメインの作業ツリーを並行調査し、その後で主要な主張を直接ウラ取りした:

1. **パターン棚卸し** — `src/` 全体を grep で走査し、横断的なイディオム（パス解決、シード設定、デバイス、config/checkpoint 読み込み、ファイルシステム IO、テンソル変換、ロギング）を抽出。
2. **タスクローカルなヘルパ** — `src/tasks/{ball_detection,base,blcs,court_detection,plcs}` を精読し、ドメイン非依存なモジュールレベル `_helper` 関数を特定。
3. **utils インベントリ + scene/tools** — 既存の `src/utils/` 公開 API を棚卸しし、`src/tennis_scene/` の重複・昇格候補を調査。

**除外:** `third_party/`（ベンダリング）、`__pycache__`、`.venv`。トップレベルの `tools/` は**未追跡のローカルツール**（プロジェクトメモによれば colab CLI）で、コミット済みのリポジトリコードではないため、共通ユーティリティ調査の対象外とした。

直接ウラ取りした項目（バイト単位の照合または grep 件数）: `_clone_sample` の重複、2つの ImageNet 正規化関数、`_resolve_device` のコピー群、`_seed_everything` のコピー群、パス解決 / ディレクトリ生成 / `to_numpy` の出現件数。

---

## きっかけとなった例

```python
# src/tasks/court_detection/models/dinov3_detr.py:29,36
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    return resolved.resolve()
```

この「相対パスをリポジトリルート基準で解決する」ニーズはコードベース全体で繰り返し現れ、しかも書き方が一貫していない:

- `src/tennis_scene/pipeline/components/gvhmr.py:143,334` — `Path(__file__).parents[3]`
- `src/tennis_scene/pipeline/orchestrator.py:309` — `cwd=str(Path(__file__).parents[3])`
- `src/tasks/blcs/generate_dataset/config.py:118` — `Path(__file__).resolve().parents[1] / "configs"`
- `src/tasks/base/training/runner.py:387` — Hydra の `to_absolute_path` をラップした `_ensure_absolute`
- `src/` 全体で計約85箇所が何らかの `Path(__file__).parents[...]` または `to_absolute_path` を使用。

脆い部分は `parents[N]` のハードコードされた深さで、新しいファイルを追加するたびにリポジトリルートまでの距離を数え直す必要がある。`src/utils/paths.py` に `PROJECT_ROOT` 定数 + `resolve_project_path()` を1つ置けば、この落とし穴を解消できる。

---

## A. ローカル再実装されている横断的な汎用パターン

| パターン | 出現箇所（確認済み） | 再利用可能か | 推奨配置先 |
|---|---|---|---|
| プロジェクトルート / パス解決 | `_resolve_project_path`（court_detection）+ `_ensure_absolute`（base runner:387）+ 約85箇所のインライン `parents[...]`/`to_absolute_path` | **可** | `src/utils/paths.py`（新規） |
| デバイス解決 | `base/inference/predictor.py:93` `_resolve_device`（正典）+ plcs コピー: `scripts/analysis/visualize_rotation_error_samples.py:46`, `scripts/generate_dataset.py:41`, `scripts/analysis/analyze_loss_dominance.py:340` | **可** | `src/utils/device.py`（新規） |
| 決定論的シード設定 | `plcs/scripts/generate_dataset.py:34` と `blcs/scripts/generate_dataset.py:46`（`_seed_everything`、**完全一致**）+ `base/training/runner.py:173`（`seed_everything` メソッド） | **可** | `src/utils/seeding.py`（新規） |
| テンソル → numpy | `base/training/lightning_module.py:150`（`_to_numpy`）、`plcs/visualization/adapters/render_inputs.py:23`（`_to_numpy`）、`court_detection/generate_dataset/annotation_session.py:436`（`to_numpy`）、加えてインライン `.detach().cpu().numpy()` | **可** | `src/utils/tensor_utils.py`（既存） |
| ディレクトリ生成（`mkdir(parents=True, exist_ok=True)` / `makedirs`） | **インライン54箇所**、ラッパなし | やや有 | `src/utils/io.py`（新規）`ensure_dir()` |
| JSON 読み書きヘルパ | private な json 保存/読み込みが複数モジュールで再実装。6つのパイプライン `Result.save()` メソッドが同一の `mkdir + json.dump(indent=2)` 本体を共有 | **可** | `src/utils/io.py`（新規） |
| config 読み込み（`OmegaConf.load` / `yaml.safe_load`） | 存在するが、呼び出し箇所ごとに文脈が異なる | **不可** | 現状維持 |
| checkpoint / state_dict の加工 | 概ね局所的。`src/utils` には既に `models/dino_backbone` 系がある。`dinov3_detr` はインラインで `torch.load` + key 展開 | 低 | 現状維持 / `src/utils/models` を拡張 |
| ロギング設定 | `basicConfig` 3箇所（すべてエントリポイント）+ 慣用的な `getLogger(__name__)` 約25箇所 | **不可** | 現状維持（慣用的） |

**重複の重大度ランキング（件数・深刻度順）:**
1. インライン `mkdir(parents=True, exist_ok=True)` — 54箇所（イディオム、ヘルパなし）。
2. パス / プロジェクトルート解決 — 約85箇所、深さの扱いが不統一。
3. `_resolve_device` — 正典1 + ドリフトした3コピー。
4. `_seed_everything` — 完全一致の2コピー。
5. `_to_numpy` / テンソル→numpy — ヘルパ3コピー + インライン。

---

## B. `src/utils/` 内部にすでにある重複

`src/utils/` 自身が完全一致の重複を抱えている。以下2つの関数は本体がバイト単位で同一で、関数名とデフォルト引数の書き方だけが異なる:

- `src/utils/video/transforms.py:35` — `normalize_tensor_imagenet(images, *, mean, std)`
- `src/utils/data/augmentation.py:59` — `normalize_tensor_images_imagenet(images, *, mean, std)`

**推奨:** 正典の実装を1つに定め、もう一方はそれに委譲（または再エクスポート）する。これにより2つの利用系統（`video/` のストリーミングと `data/` の augmentation）が乖離しなくなる。

---

## C. 昇格すべきタスクローカルなヘルパ

サブパッケージごとにグルーピング。「重複?」= 等価物が他所に既に存在するか。

### `ball_detection`
| file:line | ヘルパ | 内容 | 重複? | 推奨配置先 |
|---|---|---|---|---|
| `data/augmentation.py:44` | `denormalize_tensor_images_imagenet` | `(...,3,H,W)` に ImageNet 正規化の逆変換 | utils の `normalize_*` と対称 | `src/utils/data/augmentation.py` |
| `data/augmentation.py:32` | `normalize_frames_imagenet` | HWC numpy フレームのリストに ImageNet 正規化 | utils テンソル版の numpy 兄弟 | `src/utils/data/augmentation.py` |
| `data/augmentation.py:574,586` | `_parse_ratio_range` / `_parse_int_range` | config から2要素レンジを解析・検証 | **有** — `utils...parse_float_range` が既存、`base...:291` `_validate_range` も別コピー | `src/utils/data/augmentation.py` を拡張 |
| `data/augmentation.py:750` | `make_sample_rng` | worker 認識の決定論的なサンプル単位 RNG | 汎用的な dataloader の関心事 | `src/utils/data/`（seeding） |
| `data/augmentation.py:63` | `_resolve_border_mode` | config 文字列 → `cv2` 境界定数 | 任意の cv2 augmentation で再利用可 | `src/utils/data/augmentation.py` |

### `base`
| file:line | ヘルパ | 内容 | 重複? | 推奨配置先 |
|---|---|---|---|---|
| `data/scene_dataset.py:27` | `_load_scene_payload` | scene ディレクトリから `.npy` + `scalars.json` + `meta.json` を読み込み | プロジェクト共通の scene データセット形式で、blcs/plcs が共有 | `src/utils/data/scene_io.py`（新規） |
| `data/scene_dataset.py:291` | `_validate_range` | `(lo, hi)` が順序付き・正であることを検証 | **有** — ball_detection のコピー参照 | レンジ検証の共通ユーティリティ |
| `training/runner.py:322` | `select_devices` | config → `("gpu", N)` / `("cpu", 1)` | `_resolve_device` と同趣旨 | `src/utils/device.py` |

### `court_detection`
| file:line | ヘルパ | 内容 | 重複? | 推奨配置先 |
|---|---|---|---|---|
| `models/dinov3_detr.py:36` | `_resolve_project_path` | **今回のきっかけ** | 有（§A 参照） | `src/utils/paths.py` |
| `training/metrics.py:123` | `_heatmaps_to_pixel_coords` | argmax ヒートマップ → 逆正規化済み画素座標 | **有** — ball_detection が `inference/predictor.py:114`、`scripts/eval.py:326`、`visualization/adapters/render_inputs.py` で手動スケーリング | `src/utils/data/heatmaps.py` を拡張 |
| `inference/preprocess.py:16` | `preprocess_court_image` | 短辺リサイズ→8の倍数にスナップ→正規化→バッチ化→デバイス転送 | **有** — `data/augmentation.py:119,297` がリサイズ計算を再実装 | `src/utils/data` または `video/transforms` |
| `data/augmentation.py:35,40` | `_pil_to_tensor_image` / `_mask_pil_to_tensor` | 標準的な PIL→テンソル変換 | 3つの court データセットで共有 | `src/utils/data/augmentation.py` |

### `blcs` / `plcs`
| file:line | ヘルパ | 内容 | 重複? | 推奨配置先 |
|---|---|---|---|---|
| `blcs/data/augmentation.py:27` と `plcs/data/augmentation.py:30` | `_clone_sample` | テンソル辞書のディープクローン | **完全一致** | `src/utils/tensor_utils.py` `clone_tensor_dict()` |
| `blcs/training/losses.py:15` と `plcs/training/losses.py:118` | `trajectory_position_loss` / `position_loss` | smooth-L1 位置ロス（マスク有無） | 同一演算、マスク有り版/無し版 | 共通 `src/utils/training/losses.py` |
| `plcs/training/losses.py:159,187` | `angular_error`、`_wrapped_angle_diff` | ラジアンでのラップ角誤差 | **有** — `plcs/training/metrics.py:126-127` にインライン、numpy 版 `_angular_error_deg` が `scripts/analysis/visualize_rotation_error_samples.py:75` と `analyze_loss_dominance.py` で再導出 | `src/utils/geometry/rotation.py`（新規） |
| `plcs/training/losses.py:182` | `_normalize_vector` | 安全な L2 正規化 | 汎用的 | `src/utils/tensor_utils.py` |
| `plcs/training/losses.py:193,226,277` | `compute_joint_angles`, `compute_torsion_angles`, `signed_angle_around_axis` | 骨格 / 3D 幾何 | すでに分析スクリプトから import 済み | `src/utils/geometry/skeleton.py`（新規） |
| `plcs/utils/pose_geometry.py:14-63` | `court_position_to_world_translation`, `canonical_pose_to_world_pose`, … | 正規化コート座標 ↔ ワールドメートル系のポーズ変換 | スケール定数は `src/utils/schema/court.py` 由来。blcs の projection も同じスケールで逆正規化 | `src/utils/geometry/court_pose.py`（新規） |
| `plcs/training/metrics.py:14` | `_flatten_valid` | パディング `(B,T,D)` 上のマスク gather | 汎用的 | `src/utils/tensor_utils.py` |

**最優先の抽出候補（ショートリスト）:** `_clone_sample`、`_resolve_project_path`、`_resolve_device`、`_seed_everything`、`_heatmaps_to_pixel_coords`、`angular_error` / ラップ角クラスタ、`compute_*` 骨格幾何関数 — いずれも完全一致の重複か、すでにモジュール境界をまたいで import されているもの。

---

## D. `src/tennis_scene/` → `src/utils/` への昇格候補

| file:line | ヘルパ | 内容 | 状態 | 備考 |
|---|---|---|---|---|
| `utils/transforms.py:18` | `rotation_matrix_y(yaw)` | スカラー yaw の Y 軸回転（numpy） | 昇格 | レンダラの `_rotation_matrix_z` と統合 |
| `rendering/tennis_scene_renderer.py:151,174` | `_axis_angle_to_matrix`, `_rotation_matrix_z` | バッチ Rodrigues / Z 軸回転 | 昇格 | 同居しているが不統一な回転ヘルパ → `src/utils/geometry/rotations.py` |
| `utils/transforms.py:40,65` | `apply_plcs_transform[_batch]` | ローカル→コートの剛体変換（SMPL 頂点） | 昇格 | レンダラが `:241-246` の einsum で重複 |
| `utils/transforms.py:96,118` | `normalize_keypoints` / `denormalize_keypoints` | 画素 ↔ UV | 昇格 | ドメイン依存なし |
| `pipeline/components/{blcs,plcs,court_kp,gvhmr,ball_detection,player_association}.py` | `Result.save()` | `mkdir(parents=True) + json.dump(indent=2)` | 昇格 | **完全一致の6コピー** → `src/utils/io.py` `save_json()` |
| `pipeline/components/gvhmr.py:143,334`, `orchestrator.py:309` | `Path(__file__).parents[3]` でプロジェクトルート | サブプロセス cwd / 3rd-party import 用にリポジトリルートを特定 | 昇格 | §A と同じギャップ |

---

## 既存の正典 API（`src/utils/`）— これらは**再実装しないこと**

| モジュール | 提供内容 |
|---|---|
| `tensor_utils.py` | `masked_mean`, `normalize_padding_mask` |
| `data/augmentation` | UV / 可視性 augmentation、`normalize_tensor_images_imagenet`、`parse_float_range`、`dilate_temporal_mask` など |
| `data/heatmaps` | `generate_gaussian_heatmap[s]`、`heatmaps_to_argmax`、`heatmaps_to_soft_argmax`、`heatmaps_to_peaks` |
| `models/` | Transformer ブロック、MoE、RoPE、attention、`MLPHead`、ドメイン特化のトークン埋め込み |
| `projection/` | `Camera`、`CameraProjector`、`make_look_at_camera`、`project_points` |
| `rendering/` | `CourtRenderer`、`SkeletonRenderer`、`BallRenderer` |
| `schema/` | コート幾何（`CourtKP*`、スケール）とプレイヤーポーズスキーマ（COCO-17 / SMPL-H ジョイント、骨格、角度トリプレット） |
| `video/` | OpenCV ストリーミング: `probe_video_info`、`read_video_frame`、`OpenCVVideoFrameReader`、`iter_temporal_windows`、`iter_temporal_batches`、`PrefetchIterator`、`normalize_tensor_imagenet`、`BgrToTensorTransform` |

新しい抽出は、**最も近い既存モジュールの隣**に配置すること（例: テンソルヘルパは `tensor_utils.py`、ヒートマップ逆正規化は `data/heatmaps.py`）。自然な配置先がない場合に限り、新規モジュール（`paths.py`、`device.py`、`seeding.py`、`io.py`、`geometry/`）を作成する。

---

## 後続 PR の推奨進行順

価値対リスク比でソート（各ステップは独立した、挙動を保つ移動）:

1. **リスクゼロの重複排除:** `_clone_sample` → `tensor_utils.clone_tensor_dict`、2つの ImageNet 正規化関数を集約。（純粋な重複削除）
2. **`src/utils/paths.py`:** `PROJECT_ROOT` + `resolve_project_path()`。`_resolve_project_path`、`_ensure_absolute`、tennis_scene の `parents[3]` 箇所を移行。
3. **`src/utils/device.py`:** `BasePredictor._resolve_device` を自由関数に昇格。plcs の3コピーと `select_devices` を撤去。
4. **`src/utils/seeding.py`:** `seed_everything()` を1つに。`_seed_everything` の2コピーを撤去し、`make_sample_rng` を取り込む。
5. **`src/utils/io.py`:** `ensure_dir()`、`save_json()`、`load_json()`。6つの `Result.save()` メソッドとトラフィックの多い mkdir 箇所を移行。
6. **`heatmaps_to_pixel_coords`** を `data/heatmaps.py` に追加。court/ball の手動逆正規化箇所を撤去。
7. **`src/utils/geometry/`**（`rotation.py`、`skeleton.py`、`court_pose.py`）: 角度誤差 / 回転行列 / 骨格角 / ポーズ変換クラスタを集約。（最大。テストを充実させて最後に実施）

各ステップは独立した PR とし、移動前後の等価性テストを付けることで、挙動が保たれることを証明可能な形に保つこと。
