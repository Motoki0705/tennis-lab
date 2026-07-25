# Tennis-Lab 🎾

単眼テニス動画から、プレーヤー/ボール/コートを推定して「コート座標系の3Dシーン」に統合するための研究用モジュラーパイプライン。

## 成果

### Tennis Scene（統合3D再構成）

| 入力映像（`data/samples/tennis_clip.mp4`） | コート座標系の3Dシーン |
| :---: | :---: |
| <img src="assets/tennis_scene/tennis_clip.gif" width="400" /> | <img src="assets/tennis_scene/rich_scene_rendering.gif" width="400" /> |

- 実装/実行: [src/tennis_scene/README.md](src/tennis_scene/README.md)

### Ball Detection（2Dボール検出）

<p align="center">
  <img src="assets/ball_detection/game1_clip1_prediction.gif" width="840" />
</p>

- 実装/実行: [src/tasks/ball_detection/README.md](src/tasks/ball_detection/README.md)

### Court Detection（KP / SEG / LINE）

<p align="center">
  <img src="assets/court_detection/kp.gif" width="840" /><br/>
  <em>Keypoint：RGB + 予測キーポイント ｜ 平均ヒートマップ</em>
</p>

<p align="center">
  <img src="assets/court_detection/seg.gif" width="840" /><br/>
  <em>Segmentation：RGB ｜ セグメンテーションマップ</em>
</p>

<p align="center">
  <img src="assets/court_detection/line.gif" width="840" /><br/>
  <em>Line：RGB ｜ ラインマップ</em>
</p>

- 実装: [src/tasks/court_detection/README.md](src/tasks/court_detection/README.md)

### PLCS（プレーヤー3D位置・yaw推定）

<p align="center">
  <img src="assets/plcs/gt_vs_pred.gif" width="720" /><br/>
  <em>GT（緑）｜ Prediction（赤）</em>
</p>

- 実装/実行: [src/tasks/plcs/README.md](src/tasks/plcs/README.md)

### BLCS（ボール3D軌道推定）

<p align="center">
  <img src="assets/blcs/gt_vs_pred.gif" width="600" /><br/>
  <em>GT（緑）｜ Prediction（赤）</em>
</p>

- 実装/実行: [src/tasks/blcs/README.md](src/tasks/blcs/README.md)

### BLCSデータ生成（物理シミュレーション）

<p align="center">
  <img src="assets/blcs_sim/blcs_sim_demo.png" width="840" /><br/>
  <em>物理シミュレーションによるボール軌道・イベント・マルチカメラ観測の生成</em>
</p>

## 開発中

### Multi-object PLCS / BLCS

[PR #650](https://github.com/Motoki0705/tennis-lab/pull/650) で、複数プレーヤー・複数ボールの lifecycle を扱う生成・追跡・可視化パイプラインを開発しています。

| 複数プレーヤー（PLCS） | 複数ボール（BLCS） |
| :---: | :---: |
| <img src="assets/plcs/multi_object.gif" width="400" /> | <img src="assets/blcs/multi_object.gif" width="400" /> |
| `multi_object_lifecycle_v2 / scene_000040` | `multi_object_lifecycle_v4 / scene_000345` |

## クイックスタート

このリポジトリは [uv](https://docs.astral.sh/uv/) で依存関係を管理しています。Docker / devcontainer ベースでの開発も利用できます。

### ローカル環境（uv）

```bash
# 依存関係をインストール（仮想環境 .venv を自動作成）
uv sync --group dev

# スクリプト実行例
uv run python -m src.tasks.<task>.scripts.<entrypoint> key=value
```

> 旧 `requirements.txt` / `requirements-dev.txt` は削除済みです。依存関係の追加は `uv add <package>` で行ってください。

### Docker 環境

### 1) イメージをビルド

```bash
docker build -t tennis-lab-dev .
```

### 2) コンテナを作成して起動

```bash
mkdir -p data outputs

docker run --gpus=all -it --name tennis-lab-dev \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  tennis-lab-dev
```

既存コンテナを再利用する場合:

```bash
docker start -ai tennis-lab-dev
```

### 3) コンテナ内でスクリプト実行（共通形）

ホスト側からコンテナに入る場合:

```bash
docker exec -it tennis-lab-dev bash
```

すべてのタスクは Hydra を使って設定を管理しています。

```bash
python -m src.tasks.<task>.scripts.<entrypoint> key=value
```

### 4) まずは可視化

コマンド例はタスクごとのドキュメントに集約しています。

- [src/tasks/blcs/README.md](src/tasks/blcs/README.md)
- [src/tasks/plcs/README.md](src/tasks/plcs/README.md)
- [src/tasks/ball_detection/README.md](src/tasks/ball_detection/README.md)
- [src/tasks/court_detection/README.md](src/tasks/court_detection/README.md)

### Docker

VS Code を使う場合は `.devcontainer/devcontainer.json` を利用して devcontainer として開けます。

## ライセンス / 引用

このリポジトリは [MIT License](LICENSE) の下で公開されています。

## 構造（成果の裏側）

### どこに何があるか（タスク）

- Ball Detection: 画像上の2Dボール位置（`src/tasks/ball_detection`）
- Court Detection: 20点コートキーポイント（`src/tasks/court_detection`）
- PLCS: 2Dスケルトン → コート上3Dプレーヤー位置/yaw（`src/tasks/plcs`）
- BLCS: 2Dボール位置 → コート上3Dボール軌道（`src/tasks/blcs`）
- GVHMR: 画像列 -> 2Dスケルトン + SMPL (`third_party/GVHMR`)
- 統合: 上記をまとめて1本のパイプラインとして回す (`src/tennis_scene/README.md`)

### 典型データフロー

```
Video
  ├─ Court Detection  → court_kp (2D)
  ├─ Ball Detection   → ball_uv (2D)
  ├─ (optional) GVHMR → human_kp (2D) + SMPL (local)
  ├─ PLCS             → player_pos/yaw (3D on court)
  └─ BLCS             → ball_pos_world (3D on court)
```

### ディレクトリの役割

- `src/`: タスク実装（各タスクは `configs/` + `scripts/` + `training/` などを持つ）
- `third_party/`: 外部モジュール（例: GVHMR）。vendor codeは隔離
- `data/`: データセット/入力（大きなデータやモデルはコミットしない）
- `outputs/`: 学習ログ・チェックポイント・生成物（大きなartifactはコミットしない）
- `assets/`: README用の軽量デモ素材（GIF/PNGなど）
- `docs/`: 使い方/テスト/Dockerなどの補助ドキュメント
