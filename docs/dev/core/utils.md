# Utils モジュール概要 (`src/utils`)

`src/utils` は、プロジェクト全体で再利用される**ユーティリティ群**を提供するモジュールです。
依存関係のポリシーとして、他のアプリケーション層に依存せず、
上位レイヤー（`base`, `blcs`, `plcs` など）から参照されることを想定しています。

現在は主に、次の 2 つのサブパッケージで構成されています。

- `geometry`: コートと人体の幾何情報、カメラモデル
- `rendering`: コート・骨格・ボール・シーンの描画ユーティリティ

## 1. ディレクトリ構成

```text
src/utils
├── __init__.py          # 公開 API: geometry / rendering
├── geometry
│   ├── __init__.py      # 幾何ユーティリティのエントリポイント
│   ├── constants.py     # コート・人体・SMPL-H の定数とインデックス
│   └── court.py         # コート寸法・カメラモデル・射影ユーティリティ
└── rendering
    ├── __init__.py          # レンダラとスタイル定数のエントリポイント
    ├── ball_renderer.py     # ボールと軌跡の描画
    ├── blcs_scene_renderer.py  # BLCS 用シーンレンダラ
    ├── constants.py         # 描画スタイル定数
    ├── court_renderer.py    # コート (2D/3D) 描画
    ├── plcs_scene_renderer.py  # PLCS 用シーンレンダラ
    └── skeleton_renderer.py # 人体骨格描画
```

## 2. Geometry ユーティリティ (`src/utils/geometry`)

### 2.1. 提供される主な機能

- **コートの寸法・キーポイント定義**
  - ITF ルールに基づくコート寸法 (`COURT_LENGTH`, `SINGLES_WIDTH`, `DOUBLES_WIDTH` など)
  - フェンスやランオフ領域の寸法 (`BASELINE_CLEAR`, `SIDELINE_CLEAR`, `FENCE_HEIGHT` など)
  - 3D コートキーポイント (`COURT_KP20`) とそのインデックス/名称
    - `NUM_COURT_KP`, `COURT_KP_NAMES`, `COURT_KP_IDX`, `COURT_SKELETON`, `COURT_LINE_CONNECTIONS`
- **人体キーポイント定義 (COCO-17)**
  - `NUM_HUMAN_KP`, `COCO_KP_NAMES`, `COCO_KP_IDX`
- **SMPL-H 関連の定義**
  - `NUM_SMPLH_BODY_JOINTS`, `NUM_SMPLH_HAND_JOINTS`, `NUM_SMPLH_TOTAL_JOINTS`
  - `SMPLH_BODY_JOINT_NAMES`, `SMPLH_JOINT_IDX`, `SMPLH_TO_COCO17_MAPPING`
  - 顔キーポイントのオフセット `FACE_KEYPOINT_OFFSETS`
- **カメラモデルと射影**
  - ITF コート上の 3D キーポイントを返す `court_keypoints_3d()`
  - ピンホールカメラモデル `Camera` と、その生成関数 `make_look_at_camera`
  - 3D 点群を画像平面へ射影する `project_points`
  - フェンス上の位置からカメラ中心をサンプリングする `sample_camera_position_on_fence`

### 2.2. 典型的な利用シナリオ

- シミュレータやレンダラで、**座標系とキーポイントの定義を統一**するために使用します。
- SMPL-H ベースの 3D 姿勢推定結果を COCO-17 キーポイントへマッピングする用途など。
- コート上の任意のカメラ位置・向きをサンプリングし、レンダリングや投影に利用します。

## 3. Rendering ユーティリティ (`src/utils/rendering`)

### 3.1. 提供される主なコンポーネント

`src.utils.rendering` では、以下のクラスと定数を**公開 API**として提供します。

- スタイル定数
  - `DEFAULT_COURT_COLOR`, `DEFAULT_LINE_COLOR`, `DEFAULT_NET_COLOR`,
    `DEFAULT_BALL_COLOR`, `DEFAULT_FENCE_MARGIN`
- レンダラ
  - `CourtRenderer`: コートの 2D / 3D 描画
  - `SkeletonRenderer`: 人体骨格の描画
  - `BallRenderer`: ボールとその軌跡の描画
- シーンレンダラ
  - `PLCSSceneRenderer`: PLCS 用のシーン描画
  - `BLCSSceneRenderer`: BLCS 用のシーン描画

### 3.2. 利用例

レンダリングユーティリティは、Matplotlib などの外部ライブラリと組み合わせて利用することを想定しています。

```python
from src.utils.rendering import CourtRenderer, BallRenderer

court = CourtRenderer()
ball = BallRenderer()

fig, ax = plt.subplots()

# コートの描画
court.render_2d(ax)

# ボール軌跡の描画
ball.render_trajectory_2d(ax, positions)
```

## 4. 設計ポリシーと依存関係

- `src/utils` は、**他モジュール（`base`, `blcs`, `plcs` など）に依存しない**ことを原則とします。
- 幾何・レンダリングに関する低レベルのロジックをこのレイヤーに集約し、
  上位レイヤーではこれらを組み合わせてアプリケーションロジックを構築します。
- 公開 API (`__all__`) を通じて、利用側は `from src.utils import geometry, rendering` のように
  高レベルかつ安定したインターフェースで依存できるように設計されています。
