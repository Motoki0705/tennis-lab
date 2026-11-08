# 元動画のプレー球注釈 v3

`video_ball_annotation.v3` の正本はこのtask内の
[`contracts.py`](contracts.py)、保存・読込は [`io.py`](io.py)、利用判定は
[`usage.py`](usage.py) に置く。scene・datasetはこのAPIから読み、schemaや状態表を複製しない。
Pydanticの `VideoBallAnnotation.model_json_schema()` で機械可読schemaを取得できる。

対象はカメラが向くコートでプレーしている**生きた1球**。
`target.target_semantics="active_ball"` を必須とし、全ボール検出とは区別する。
手持ち球・予備球・隣接コートの球が見えても、プレー中の球がなければ `no_active_ball`。
それらの球の座標を対象球へ流用しない。`track_id` はこの注釈scopeの対象を識別する文字列で、
複数球の座標配列やモデルの存在scoreは持たない。

## 入力・座標の契約

| field | 意味・検証 |
|---|---|
| `schema_version` | `video_ball_annotation.v3` を明記する。欠落・v2の自動読替えを拒否 |
| `source` / `VideoIdentity` | 元動画の `media_sha256`・bytes数・file名、`recording_id`・`camera_id`、native幅/高さ、全frame数、`time_base`・`nominal_fps`、rotation=0 |
| `target` / `BallTarget` | 必須の `target_semantics=active_ball`、label=tennis_ball、`track_id` |
| `frame_range` | 元動画のpresentation frame番号による半開区間。全動画でも部分区間でもよい |
| `frames` / `VideoBallFrame` | 区間内の全frameを元順序で1回ずつ保持。`frame_index` は元動画の0始まり番号、`pts` は元streamの実整数PTS |
| `coordinate_system` | 必須の `native_pixels_top_left_endpoint`。左上原点、x右向き・y下向き、native pixel中心座標 |
| `position` / `PixelPosition` | `null` または有限な `x,y` と `provenance`。`0≤x≤width−1, 0≤y≤height−1` |
| `occlusion` | `not_occluded / occluded / unknown / not_applicable`。座標の有無・由来と独立に記録 |
| `position.provenance` | `kind=observed / interpolated / estimated / occlusion_estimated` と同一元動画/cameraの `support_frames` |
| `break_before`, `notes` | 連続性の切れ目と自由記述を保持。breakをhit/bounce等へ推測しない。既定値はfalseと空文字 |

`recording_id` は全camera・関連clipで共有する元収録group。
動画bytesのidentityとは別であり、下流がsplitを決めるための入力である。
入力に切出し済み動画を使う場合、frame番号・PTSはそのファイルを基準にし、収録全体との対応は
登録側のmanifestが保持する。file名からcameraや収録groupを推測しない。

PTSは厳密に増加し、`pts * Fraction(time_base)` が元stream時刻になる。
開始PTSは0でなくても負でもよく、VFRの間隔を保つ。FPSからPTSを再生成しない。
回転済み・resize/crop後の座標をnativeへ偽装しない。範囲・frame数・順序・PTSの欠落や重複、
余分なfield、型の暗黙変換、NaN/Infを拒否する。

**画像外の推定座標は保存しない設計**とする。画像外への外挿を画像内の利用可能点に混ぜないため、
writer/readerとも範囲外座標を拒否し、clampや座標の黙示削除をしない。
画面外と判断できた対象球は `out_of_frame, position=null` で保存する。
このため全 `position_usable` は画像内座標である。画像外外挿の保存を将来追加する場合は、
座標領域と利用判定を明示した契約変更が必要になる。
正規化座標は重複保存しない。必要な下流はpixelから明示的に変換する。

## 状態・由来と利用判定

`state`、`occlusion`、`position.provenance.kind` は別fieldである。
以下の「処理済み」は注釈判断が保存された意味で、各frameを独立に画像確認したという申告ではない。
観測以外にも前後文脈からの補完・推測を許し、追加QAや補間秒数上限をこのschemaで要求しない。

| 意味 | state | occlusion | position / kind | 処理済み | 位置利用可 | 直接観測 | 推定 | active_ball負例候補 |
|---|---|---|---|---|---|---|---|---|
| 直接見えた点 | in_frame | not_occluded | 座標 / observed | yes | yes | yes | no | no |
| 補間した点 | in_frame | not_occluded / occluded / unknown | 座標 / interpolated | yes | yes | no | yes | no |
| 文脈等から推定した点 | in_frame | not_occluded / occluded / unknown | 座標 / estimated | yes | yes | no | yes | no |
| 遮蔽位置の推定 | in_frame | occluded | 座標 / occlusion_estimated | yes | yes | no | yes | no |
| 遮蔽、位置は決められない | in_frame | occluded | null | yes | no | no | no | no |
| 生きた球は画像外 | out_of_frame | not_applicable | null | yes | no | no | no | yes |
| プレー中の対象球なし | no_active_ball | not_applicable | null | yes | no | no | no | yes |
| 処理しても決められない | unknown | unknown | null | yes | no | no | no | no |
| まだ判断していない | unprocessed | unknown | null | no | no | no | no | no |

遮蔽と推定座標の有無は直交する。遮蔽なし/不明の推定、遮蔽ありの補間・一般推定・遮蔽推定、
遮蔽ありで座標なしを表現できる。`observed` は当該画像で位置が直接見えた意味なので遮蔽なしを要求する。
画像内・座標なしを遮蔽以外の理由で確定できない場合は `unknown` とし、別の状態名に逃がさない。

`interpolated` は元frameを挟む2つの `support_frames` を必須とする。
`observed` のsupportは空。一般推定・遮蔽推定のsupportは任意で、記録する場合は重複のない昇順とする。
全supportは同じ元動画の実在frameを指し、注釈scope外の同camera文脈も参照できる。
補間supportを機械的な線形補間や独立目視の証明とは扱わない。補完方法の説明は `notes` に残せる。
重み・confidence・平滑化規則はここへ加えず、scene/datasetが由来を保持したまま明示する。

各行には `is_processed`、`is_unprocessed`、`is_final_unknown`、`is_position_usable`、
`is_position_unavailable`、`is_direct_observation`、`is_estimated`、
`is_active_ball_negative_candidate(..., target_semantics="active_ball")` を使う。
負例判定は他のtarget semanticsを拒否する。
負例「候補」はactive_ball教師向けであり、画像内に球が一つもないという意味や、
全ボール検出器へ負例として採用する許可ではない。遮蔽、unknown、未処理は負例候補にしない。

`usage_masks(annotation)` はこれら8判定を全frame順のbool tupleとして返す。
`summarize_annotation(annotation)` は同名maskの件数、total、in_frame、out_of_frame、
no_active_ball、occluded_with_position / occluded_without_positionを返す。

- `position_unavailable = 処理済み ∧ 位置利用不可`。座標なし遮蔽・画面外・no_active_ball・最終unknownを含む。
- `total = position_usable + position_unavailable + unprocessed`。直接観測と推定の和が位置利用可。
- `complete = (unprocessed == 0)`。最終unknownが残っていても完成できる。
- 親の実動画受入に使う81件/25件は `position_unavailable` / `final_unknown` を全cameraで合計する。
  未処理があれば未完走として別に報告する。API自身は閾値やcamera別上限を設けない。

## 保存・再読込

公開APIは `src.tasks.ball_detection.annotation.video` からimportする。

```python
from src.tasks.ball_detection.annotation.video import (
    read_annotation,
    summarize_annotation,
    usage_masks,
    write_annotation,
)

# expected_input: 登録時に固定したVideoAnnotationInput（元動画/対象/全frameと実PTS）
# layout: 共通AnnotationLayout。annotation: 検証済みVideoBallAnnotation
reference = write_annotation(
    layout, new_revision_path, annotation, expected_input=expected_input
)
saved = read_annotation(new_revision_path, expected_input=expected_input)
masks = usage_masks(saved)
summary = summarize_annotation(saved)
```

writerは共通の不変artifact保存を再利用する。同じbytesの再送は同じ参照を返し、
異なる内容は旧v2を含め上書きしない。途中保存も新しいrevision pathへ書く。
`model_copy()` などで検証を迂回した変更も、保存前に全体を再検証する。
revision選択・CAS・workerの実行制御はこのAPIの責務外である。

readerは保存済み内容の構文・状態・座標・timeline整合性を毎回検証する。
`expected_input` を渡すと元動画metadata全体、収録/camera、対象、対象区間と全frame/PTSの一致を要求する。
scene/datasetへの接続時は登録側の固定入力から作ったexpected_inputを渡す。
同じ注釈から取り直した `annotation.input_identity` だけでは、外部入力との照合にならない。
省略は単独成果物の閲覧・集計向けの自己整合性検証である。
元動画の実bytes・復号結果との照合は入力登録側の責務で、readerは動画ファイルの存在を要求しない。
保存した `ArtifactRef` のbytes検査は共通 `verify_reference()` と組み合わせる。

`require_complete=True` は未処理があれば拒否する。既定falseでは途中保存を読める。
最終unknownはどちらでも読める。完成判定を「unknownが0」と定義し直さない。

## v2との対応・移行境界

現行v2の意味は既存readerを照合した。
[`scene`](../../../../tennis_scene/pipeline/imports/ball_annotations.py) はobservedだけを観測として採用し、
[`meiji`](../../generate_dataset/frame_store/sources/meiji.py) は点種を保持、
[`statistics`](../../dataset_statistics/inputs.py) は補間support・notesを読む。
v3の利用判定を旧readerへ付け替える変更はここでは行わない。

| v2の情報 | v3で保持できる対応 | v2だけでは分からないもの・条件 |
|---|---|---|
| source.sha256 / file_name / width / height / frame_count / time_base / fps | sourceのmedia_sha256等へ保持、FPSは有理数文字列へ | bytes数・camera・元収録groupがなければ元動画/登録資料から別途確認 |
| frame_index / pts | 同じ元ファイルを基準にそのまま保持 | FPSから欠けたPTSを捏造しない。scene旧readerがPTSを読まなくてもv3では必須 |
| target.track_id、label | labelとtrack_idを保持。整数IDは文字列へ | 「プレー中の1球」か全ボールかはIDだけでは判別不可。active_ballの意味を別途確認 |
| observed / visible + center_px | in_frame / not_occluded / observed | 当時の観測申告を保持。認識精度が再検証済みという意味ではない |
| interpolated / not_independently_visible + center_px | in_frame / **unknown occlusion** / interpolated | not_independently_visibleから遮蔽の有無を断定しない。元frameを挟むsupport 2件が必要で、欠落なら再確認 |
| occlusion_estimated / occluded + center_px | in_frame / occluded / occlusion_estimated | 遮蔽の判断と推定座標の由来を両方残す |
| unresolved / unknown + null | 注釈処理の完了を確認できた場合、unknown / unknown / null | out_of_frame / no_active_ball / 座標なし遮蔽を推測しない。v2には未処理との独立fieldもなく、statusだけで完了を保証しない |
| center_normalized = x_px/width, y_px/height | 合意誤差内でcenter_pxと照合後、pixelだけを保持 | v3はpixel端点範囲。旧正規化値をx/(width−1)と読み替えない。旧v2が許したwidth−1超のsubpixel点は再確認し、黙ってclampしない |
| support_frames / break_before / notes | support / break_before / notesへ保持 | 不足supportを推測しない。旧review・元JSON・経緯は不変の元artifactとして残す |
| 旧snapshotのgroup/split | 将来の移行/公開時に照合・維持 | 注釈schemaだけから既存の評価用groupをtrainへ移さない |
| out_of_frame / no_active_ball / unprocessed、座標なし遮蔽 | v3で新たに表現可能 | v2の4statusから自動復元できない。再注釈/外部の確かな記録が必要 |

この実装にはv2自動変換APIも既存データ変換の実行も含めない。
v3 readerへv2を渡すと拒否する。既存v2 JSON・snapshot・readerはそのまま残す。
移行は後続 #1714 で元artifactを保存したまま別のv3 artifactを作り、保持情報と再確認情報、
座標/PTS・由来・group/splitを照合する。scene接続は #1711、dataset公開は #1712 が担当する。

## 検証

```bash
.venv/bin/python -m pytest -n 0 tests/unit/tasks/ball_detection/annotation/video
```

非対称なcamera/解像度・非zero元frame区間・可変PTSの10行に、観測・補間・一般推定・
遮蔽推定・遮蔽あり補間・座標なし遮蔽・画面外・対象なし・最終unknown・未処理を混在させ、
maskと件数を手計算と照合する。入力差替え拒否、不正状態/座標/由来、実JSONの不変保存と
再読込、旧v2の4status対応と上書き拒否を含む。v2サンプルはmetadata/label解析用で動画復号を行わない。
実GPT認識・実動画移行・scene三角測量・dataset公開の成立を、この契約テストの結果で代用しない。
