# 人間提供の評価用正解 reader

`tennis_lab_ball3d_user_reference.v1` の統合 JSON を、再構成 pipeline と独立して読みます。
正解は前半の調整・採点・評価可視化にだけ使います。検出結果、heatmap evidence、
再構成方式の入力への変換や接続は提供しません。採点処理への接続は #1556 の担当範囲です。

## 公開 API

`src.tennis_scene.ground_truth` から import します。

| API | 契約 |
|---|---|
| `ReferenceIdentity(clip_id, camera_ids, num_frames, fps, source_size_wh, video_sha256)` | 利用側が信頼済み入力から用意する identity。camera は順序付き tuple、サイズは `(width, height)`、hash は camera ID ごとの SHA-256 |
| `load_user_reference(path: Path, *, expected: ReferenceIdentity, expected_sha256: str) -> UserReference` | JSON の hash、schema、identity、元 CSV の hash・サイズと構造を検証して読み込む |
| `sha256_file(path: Path) -> str` | ファイルの実バイトを SHA-256 で検査する。動画デコードはしない |
| `UserReference.position(camera_id, frame_index) -> ReferencePosition` | camera ID と元動画の 0 始まり frame 番号で 1 行を取得する |
| `UserReference.select(*, camera_ids, frame_indices) -> ReferenceSelection` | 指定順の camera/frame 軸で UV・採点 mask・reference kind を取得する |

`ReferencePosition` は `camera_id`, `frame_index`, `source_status`, `reference_kind`,
`valid_for_scoring`, `uv_px`, `source_file`, `estimation` を持ちます。座標なしは `None`。
推定 method、quality、spread、event や補間端点などの元文字列は `estimation` にそのまま残します。

`ReferenceSelection` は `camera_ids`, `frame_indices` と、`uv_px` `(V,T,2)` float64、
`valid_for_scoring` `(V,T)` bool、`reference_kind` `(V,T)` 文字列の読み取り専用配列です。
座標なしを配列に表すときだけ `NaN` とし、同じ場所の mask は必ず false です。
数値の 0 や finite 判定から採点可否を推測せず、mask を使ってください。
軸は明示指定が必須で、重複・空の指定、未知の camera、範囲外・負・bool の frame は拒否します。

`UserReference.positions` は `(camera_id, frame_index)` を key とする読み取り専用 mapping。
`events` と `estimation_segments` は元行順の tuple で、各行の全 field を保持します。
camera ごとのイベント・note・source を残し、物理イベントへの集約や重複除去はしません。
`metadata` は positions/events/segments 以外の全 field（provenance、counts、notes 等）を保持します。
これらの nested mapping/sequence も読み取り専用です。`sha256` は読み込んだ JSON の hash です。

## 入力契約

JSON には v1 の全 field が必要です。未知 field、重複 JSON key、未知 schema・kind、
重複 camera/frame、1 行でも欠けた timeline、非有限値、型の暗黙変換を拒否します。
全 camera が同じ frame 数・FPS・native 寸法を持つ契約です。

identity は clip ID、camera の順序、frame 数、native 寸法、camera ごとの動画 hash を照合します。
FPS の許容差は pipeline の `ClipSource` と同じ絶対差 `1e-6` で、frame の再対応付けはしません。
利用側は `expected` を正解 JSON 自身から作らず、固定入力や検証済み動画から渡します。
動画の実バイトの検査は利用側で `sha256_file` を使い、固定入力の hash と照合してください。
`expected_sha256` も信頼済みの保存記録から渡します。

`provenance.kind` は `user_supplied_reference`。`source_files` に記録された CSV は
JSON の directory からの相対パスで存在する必要があり、全ファイルのバイト数と hash を毎回照合します。
外部へのパス、重複 filename、未登録の `source_file` を拒否します。
CSV の意味的な再統合は reader では行いません。これは元 JSON の作成・独立照合の責務です。

| reference_kind | source_status | valid_for_scoring | uv_px |
|---|---|---|---|
| `observed` | `observed` | true | native pixel 座標 |
| `estimated` | `unknown` | true | ユーザーが承認した推定座標 |
| `excluded_extrapolation` | `unknown` | false | null |
| `occluded` | `occluded` | false | null |
| `unknown` | `unknown` | false | null |

採点可能な座標は有限な数値 2 個で、`0 <= u <= width-1`, `0 <= v <= height-1` が必要です。
観測・推定の偽装や採点可否との矛盾を拒否します。推定には元の推定 provenance が必要です。
`provenance.exclusion` の区間と `excluded_extrapolation` および `extrapolated_unverified` flag は
厳密に対応しなければなりません。除外区間を clip 固有の番号としてコードへ埋め込んではいません。
元の遮蔽・不明と、ユーザーが除外した外挿は異なる kind のまま保持します。
camera ごとの件数も行データと照合します。推定区間の frame 範囲・長さ・camera/gap ID の一意性を検証し、
区間内の全点が `estimated` であることを要求します。

欠けた値の補完、座標の再推定、クリッピング、別 clip への fallback はありません。
契約違反は `ValueError`、存在しないファイル等は該当する `OSError` として停止します。

## 利用例

以下の `source` は評価用固定入力の検証済み `ClipSource`、`video_paths` は camera ID と
ローカル動画の明示的な対応、`ground_truth_path` と `pinned_json_sha256` は保存済み参照の場所と hash です。

```python
from src.tennis_scene.ground_truth import (
    ReferenceIdentity,
    load_user_reference,
    sha256_file,
)

for video in source.videos:
    if sha256_file(video_paths[video.camera_id]) != video.sha256:
        raise ValueError(f"Video bytes mismatch: {video.camera_id}")

expected = ReferenceIdentity(
    clip_id=evaluation_clip_id,  # 評価 bundle が宣言する clip ID を明示
    camera_ids=source.camera_ids,
    num_frames=source.num_frames,
    fps=source.fps,
    source_size_wh=source.size,
    video_sha256={v.camera_id: v.sha256 for v in source.videos},
)
truth = load_user_reference(
    ground_truth_path,
    expected=expected,
    expected_sha256=pinned_json_sha256,
)
reference = truth.select(camera_ids=source.camera_ids, frame_indices=range(505, 1010))
uv = reference.uv_px
score_mask = reference.valid_for_scoring
direct_mask = score_mask & (reference.reference_kind == "observed")
estimated_mask = score_mask & (reference.reference_kind == "estimated")
# 採点には score_mask 全体を使用。direct/estimated は出自別の内訳に使用。
```

clip_000 の bundle は公開の評価 clip ID と元 store 内部の `standalone` ID を別に保持しています。
reader は別名を推測しません。検証済み bundle の評価 clip ID を明示して渡してください。

## 検証

CI は `tests/unit/tennis_scene/ground_truth/` の小さな独立 JSON と source fixture だけを使います。
実動画・個人の outputs/data は不要です。非対称な camera/frame/座標を手計算で照合し、
誤った軸対応、推定の脱落、外挿の採用、identity・source hash の混同を検出します。

```bash
.venv/bin/python -m pytest -n 0 tests/unit/tennis_scene/ground_truth --collect-only
.venv/bin/python -m pytest -n 0 tests/unit/tennis_scene/ground_truth --durations=0 --durations-min=0
```

実正本と元 CSV・動画の独立照合結果、件数と実測時間は #1667 の PR に記録します。
