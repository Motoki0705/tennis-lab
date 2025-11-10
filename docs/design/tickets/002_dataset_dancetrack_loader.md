# チケット: DanceTrack Dataset 実装（v2 transforms）

## 背景/目的
DanceTrack シーケンスから `[T,3,H,W]` のテンソルと bbox/ID ターゲットを読み出す Dataset を実装する。前処理は torchvision.transforms.v2 に限定する。

## 対象
- `src/datasets/dancetrack.py`

## タスク
- [ ] ルート `third_party/DanceTrack/dancetrack` 構造を読み取り、sequence 単位のフレームパスとアノテーションをインデックス化。
- [ ] `torchvision.io.read_image` + `transforms.v2`（`Resize`, `ConvertImageDtype`, `Normalize`, `RandomHorizontalFlip`）で前処理。
- [ ] bbox/ID を `TargetFrame`（center/size/track_id/conf）構造に整形。
- [ ] `__getitem__` で可変長 window を返すための `window_sampler` を受け取り、フレームとターゲット列を返却。
- [ ] 例外時（欠損フレーム等）のフォールバック（スキップ/空ターゲット）を明確化。
- [ ] 型ヒント・ドキュメントを整備し ruff/mypy を通す。

## 受け入れ基準（DoD）
- [ ] 代表的なシーケンスで 1 サンプルが `[T,3,H,W]` とターゲット列を返す（手動/簡易テスト）。
- [ ] transforms/v2 のみで完結（独自 transform なし）。
- [ ] `uv run pytest -k dancetrack` のスモークが通る（後続のDataModuleテストが参照）。
