# Vendored GVHMR (inference-only)

`third_party/GVHMR` (fork of [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR),
vendored from `Motoki0705/GVHMR` @ `900e8c4471e80bb6db5cb14c404c1a0766aba5f4`)
の推論に必要なコードを、メインの `.venv` で動くように移植したパッケージ。

## ライセンス・出典

GVHMR由来のコードは同梱の [LICENSE](LICENSE) に従う。
教育・研究・非営利目的に限り使用・複製・改変・配布でき、改変物もソース公開と
商用利用禁止、著作権等の表示の保持が必要になる。ルートのApache-2.0の対象には含まれない。
LICENSEは [zju3dv/GVHMR の固定commit](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/LICENSE)
から全文を変更せず取得した。

内包する第三者由来部分には、それぞれの条件もある。

- Facebook / QuaterNet由来のquaternion演算: [CC BY-NC 4.0全文](LICENSE.QuaterNet)。
- OpenMMLab / ViTPose由来部分: [Apache-2.0全文](LICENSE.ViTPose)。
- HMR2 / 4D-Humans由来部分: [MIT全文](LICENSE.HMR2)。

ViTPoseのheatmap headとmodel builderは、公式Apache-2.0版を出典に置き換えた。
ファイルごとの出典・条件・未解決事項は [ルートの一覧](../../../../LICENSES.md) を参照する。

## third_party 版からの主な変更

- `pytorch3d.transforms` への依存を `src/utils/geometry/rotation_conversions`
  （純 torch 実装）に置換。コンパイル済み pytorch3d は不要。
- hydra / MainStore によるインスタンス化を削除し、明示的なファクトリ
  （`pipeline.build_gvhmr_demo_model`）に置換。設定値は SIGA24 released ckpt
  の demo 構成を固定で採用。
- 学習専用コード（loss、データセット、augmentation、wis3d、DPVO/SLAM）は移植しない。
- `hmr4d.utils.matrix` / `geo_transform` / `hmr_global` / `net_utils` などは
  推論パスで使用される関数のみ抽出。
- ViT backbone は HMR2 と ViTPose で同一実装（`hmr2/vit.py`）を共有
  （元リポジトリでは実質同一のコードが 2 箇所にあった）。
- 動画 I/O は `src/utils/video` を利用。
- torch 2.9 対応: `torch.cuda.amp.autocast` → `torch.amp.autocast("cuda")`、
  `torch.load(..., weights_only=False)` を明示。

## モデル資産

- regressor等のSMPL補助資産は本repoに同梱せず、利用者が取得して `ckpt/body_models/gvhmr/` に置く。
  [取得・検証手順](../../README.md#smpl補助資産の取得)を参照する。
- 学習済み重みは `ckpt/`（`third_party/GVHMR/inputs/checkpoints` への symlink）から読む。
- **SMPL-X 本体（`SMPLX_NEUTRAL.npz`）はライセンス登録が必要**。
  https://smpl-x.is.tue.mpg.de/ から取得し、`ckpt/body_models/smplx/` に配置する。

## 型チェック

研究コードの移植のため、`pyproject.toml` で `src.submodules.vendor.*` は
mypy の strict ルールを緩和している。新規コードをここに追加しないこと
（typed なラッパーは `src/submodules/models/` に置く）。
