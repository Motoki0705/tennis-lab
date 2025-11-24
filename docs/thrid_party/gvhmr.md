# GVHMR 連携メモ（API 化のベース情報）

このドキュメントは、`third_party/GVHMR` リポジトリを **テニスシステムから API として呼び出す** ことを想定したときのベース情報をまとめたメモ。

- 元コード: `third_party/GVHMR/*`
- 論文: *World-Grounded Human Motion Recovery via Gravity-View Coordinates* (SIGGRAPH Asia 2024)
- ライセンス: 詳細は `third_party/GVHMR/LICENSE` を参照

---

## 1. リポジトリ構成（tennis-lab 観点）

GVHMR 側は独立した Python プロジェクトになっており、`third_party/GVHMR` 直下に `pyproject.toml` / `requirements.txt` などを持つ。

- **トップレベル**
  - `README.md` : インストール・Quick Start・論文リンク
  - `requirements.txt` / `pyproject.toml` : 依存関係（PyTorch, PyTorch-Lightning, Hydra, PyTorch3D, ViTPose, DPVO など）
  - `hmr4d/` : 実際のモデル・前処理・ユーティリティの実装
  - `tools/` : デモ・学習・評価の CLI スクリプト

- **GVHMR 関連の主要ファイル**
  - `hmr4d/model/gvhmr/gvhmr_pl.py`  
    - 学習用の `GvhmrPL (pl.LightningModule)` 実装
  - `hmr4d/model/gvhmr/gvhmr_pl_demo.py`  
    - 推論用の軽量 PL モジュール `DemoPL` と `DemoPL.predict` メソッド
  - `hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py`  
    - `Pipeline(nn.Module)`
    - 2D キーポイント + カメラ情報 + 画像特徴から SMPLX パラメータ列を推論する本体
  - `hmr4d/model/gvhmr/utils/endecoder.py`  
    - `EnDecoder` クラス
    - SMPLX パラメータ <-> 低次元表現の encode/decode、および FK などのユーティリティ
  - `hmr4d/configs/demo.yaml`  
    - Demo 用の Hydra 設定（`model: gvhmr/gvhmr_pl_demo` など）
  - `hmr4d/configs/demo_gvhmr_model/siga24_release.yaml`  
    - リリース済みチェックポイント（`gvhmr_siga24_release.ckpt`）を使ったモデル設定
  - `hmr4d/build_gvhmr.py`  
    - `build_gvhmr_demo()` で Demo 用 PL モジュールを組み立てるヘルパー

---

## 2. 推論エントリポイント

### 2.1 CLI レベル: `tools/demo/demo.py`

- GVHMR プロジェクトの README にある Quick Start
  - `python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s`
- 主な流れ：
  1. `parse_args_to_cfg()` で Hydra 設定 (`demo.yaml`) を構築
  2. `run_preprocess(cfg)` で **トラッキング・2D 姿勢・画像特徴・VO** などを事前計算
  3. `load_data_dict(cfg)` で DemoPL に渡すための `data` dict を構築
  4. `DemoPL` を Hydra で instantiate + `load_pretrained_model()` で ckpt 読み込み
  5. `model.predict(data, static_cam=cfg.static_cam)` で SMPLX シーケンスを推論

この CLI は **動画パス -> 推論結果 + 可視化動画** までを一括で行う高レベル入口。

### 2.2 Python レベル: `build_gvhmr_demo()`

- ファイル: `hmr4d/build_gvhmr.py`
  - `OmegaConf.load(PROJ_ROOT / "hmr4d/configs/demo_gvhmr_model/siga24_release.yaml")`
  - `cfg.model` を `hydra.utils.instantiate(..., _recursive_=False)` して `DemoPL` を生成
  - `gvhmr_demo_pl.load_pretrained_model(PROJ_ROOT / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt")`
  - `.eval()` 済みの `DemoPL` を返す

**テニスシステムから API として使う場合、最小限のラッパはこの関数をそのまま呼ぶ形が有力。**

---

## 3. DemoPL の推論インターフェース

ファイル: `hmr4d/model/gvhmr/gvhmr_pl_demo.py`

```python
class DemoPL(pl.LightningModule):
    def __init__(self, pipeline): ...

    @torch.no_grad()
    def predict(self, data, static_cam=False):
        ...
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.pipeline.forward(batch, train=False, postproc=True, static_cam=static_cam)
        ...
        return pred
```

### 3.1 入力 `data` の想定構造

`DemoPL.predict` の docstring では次のように記載されている。

- `length`: `int` or `Tensor` (フレーム長 F)
- `kp2d`: `(F, 3)` とコメントされているが、内部で呼び出す `normalize_kp2d` の docstring では `(B, L, J, 3)` を前提としており、
  実際の shape は **実データで確認する必要がある**（TODO）。
- `bbx_xys`: `(F, 3)`  
  - [cx, cy, s] 形式のバウンディングボックス (中心 + スケール)
- `K_fullimg`: `(F, 3, 3)`  
  - 画像全体のカメラ内部パラメータ
- `cam_angvel`: `(F, 3)` or `(F, 6)`  
  - カメラの角速度（`Pipeline` 側では `(B, L, 6)` を想定）
- `f_imgseq`: `(F, C, H, W)`  
  - 画像から抽出された特徴（VitPose とは別の CNN/ViT ベース特徴）

テニス側から API 化する際には、まず **GVHMR のデモ CLI で実際に保存されるテンソル shape** を確認し、それに合わせて input を準備するのが安全。

### 3.2 出力 `pred` の構造

- `pred["smpl_params_global"]` : dict
  - SMPLX パラメータ（重心座標系: GVHMR 独自の "gravity-view" 系）
  - 例: `global_orient`, `body_pose`, `betas`, `transl` など（各キーは `(F, *)` 系列）
- `pred["smpl_params_incam"]` : dict
  - カメラ座標系での SMPLX パラメータ
- `pred["K_fullimg"]` : `(F, 3, 3)`
- `pred["net_outputs"]` : `Pipeline.forward` からの中間出力一式

**テニスシステム視点では、`smpl_params_global` を 3D 人体ポーズ列として扱い、座標系変換だけ自前の表現に合わせればよい。**

---

## 4. デモ前処理パイプライン（tools/demo/demo.py）

`tools/demo/demo.py` は、動画ファイルから `DemoPL.predict` 入力 `data` を構築するための前処理パイプラインを持つ。

- **トラッキング**
  - `Tracker().get_one_track(video_path)` で 1 人分の bbox トラックを取得
  - `get_bbx_xys_from_xyxy` で `(F, 4)` bbox -> `(F, 3)` [cx, cy, s] へ変換
- **2D 姿勢推定 (VitPose)**
  - `VitPoseExtractor().extract(video_path, bbx_xys)` -> `vitpose`
- **画像特徴抽出**
  - `Extractor().extract_video_features(video_path, bbx_xys)` -> `vit_features`
- **Visual Odometry**
  - 静止カメラの場合: `-s/--static_cam` で VO をスキップ
  - 動くカメラの場合: `SimpleVO` もしくは DPVO ベースで回転を推定
- **カメラ行列・テンソル化**
  - `estimate_K`, `create_camera_sensor` などから `K_fullimg` を構築
- **`load_data_dict(cfg)`**
  - 上記の結果をまとめて `DemoPL.predict` に渡せる `data` dict を作る

テニスシステムで同様の前処理を流用するか、自前パイプラインを使って同じフォーマットの `data` を用意するかを設計時に選択する。

---

## 5. テニスシステムからの API 化パターン

### 5.1 パターン A: "動画パス -> SMPL シーケンス" の黒箱 API

- tennis-lab からは **動画ファイルパス** だけを渡し、GVHMR 側で
  1. `build_gvhmr_demo()` で `DemoPL` を構築
  2. `tools/demo/demo.py` 相当の前処理を Python 関数として呼び出し
  3. `DemoPL.predict(data, static_cam=...)` を実行
- メリット:
  - 実装工数が最小
  - GVHMR 側の改良をそのまま取り込みやすい
- デメリット:
  - テニス側の既存パイプライン（トラッキング/カメラ推定など）と二重になりやすい
  - パフォーマンスや入出力制御を細かく調整しづらい

### 5.2 パターン B: "2D キーポイント + カメラ情報 -> SMPL シーケンス" API

- tennis-lab 側で以下を準備して GVHMR に渡す：
  - 2D キーポイント `kp2d`
  - bbox `bbx_xys`
  - カメラ内部パラメータ `K_fullimg`
  - カメラ角速度 `cam_angvel`
  - 画像特徴 `f_imgseq`
- GVHMR 側では `DemoPL.predict` に直接 `data` を渡す形に単純化。
- メリット:
  - テニス側の既存コンポーネント（トラッカー・カメラ推定器など）を再利用しやすい
  - 将来的に multi-cam との整合や、独自の前処理を組み込みやすい
- デメリット:
  - `data` 各フィールドの正確な shape / 座標系の理解が必須
  - tennis-lab から見た責務境界がやや複雑になる

### 5.3 座標系まわりの注意

- `Pipeline` 内では、
  - カメラ座標系 (`pred_smpl_params_incam`)
  - 重力方向を考慮した world 系 (`pred_smpl_params_global`)
  の両方を扱う。
- 最終的にテニスシステムの 3D 表現（コート座標・ワールド座標など）に合わせるには、
  - GVHMR の world 系 (`ay` / `ayfz` など) からテニス側座標系への変換を設計する必要がある（要調査）。

---

## 6. 今後の実装タスク候補（メモ）

- **T1: `build_gvhmr_demo()` をラップする Python API**
  - 例: `src/third_party/gvhmr_api.py` に
    - `load_gvhmr_model(device)`
    - `run_gvhmr_on_video(video_path, static_cam: bool = False, ...)`
  - などのヘルパーを用意する
- **T2: Demo の前処理を tennis-lab 流に差し替えたパス**
  - 既存のテニス用トラッキング・カメラ推定と GVHMR の期待フォーマットのマッピングを整理
- **T3: 出力 SMPLX パラメータをテニス座標系に埋め込むコンバータ**
  - コート座標系・プレーヤー ID などと紐づけるための中間レイヤを設計

このドキュメントは「GVHMR をテニスシステムからどうラップするか」を検討する際の出発点として運用し、
実装が進んだら具体的な API 仕様やサンプルコードを追記する想定。
