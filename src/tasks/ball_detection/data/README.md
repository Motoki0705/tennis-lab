# Ball Detection のデータセット体系

学習・評価・レビューの現行入力は、`data/ball_detection/<version>` の統一ball frame storeです。
増分公開されたversionではcomponentを分け、`data/ball_detection/<version>/frames/`を開きます。
snapshotの公開・導入・復旧は[増分公開API](../generate_dataset/frame_store/publishing/README.md)を参照してください。
既存snapshotから別versionの可搬subsetを作る場合は[配布store API](distribution/README.md)を使います。
RGB shardを複製・hardlinkせずに配布ファイルと全量の内容manifestを準備できます。
現物のversion・件数・source/split内訳はレビューUIの「体系・内訳を見る」で確認します。
その画面は検証済みindexから集計するため、READMEへ件数を重複管理しません。

| 系列 | コード対応・保存先 | 入力と教師 | 用途・制限 |
|---|---|---|---|
| 統一ball frame store | `ball_detection_frames.v1` / `v2`。`BallFrameStore` がschemaを検証。通常学習の既定は `ball-mix-v2` | 1カメラの連続RGB JPEGと、point_kind付き2D ball注釈。全フレームを保存 | 学習・validation・test・データ品質レビュー。使用source・混合比・教師方針は[学習data設定](../configs/data/rgb_sequence.yaml)が正本 |
| 元動画active_ball snapshot | `ball_detection_frames.v4`。注釈schema v3やsnapshotの版名とは独立 | native RGB JPEG、元v3全体、位置あり推定を含む明示教師maskと均一重み | `ActiveBallWindowDataset`で読み、従来の汎用教師readerとの対象意味の混在を拒否 |
| 派生player pose store | `ball_detection_player_poses.v1`。`data/ball_detection/*/manifest.json` と保存済みcampaign artifact | ball storeのRGBからモデル推定した2D pose・人物枠・クリップ内選手/raw ID | ボールレビューの補助表示。ボール教師や独立したpose GTではない。manifestの元ball store hashとclipの対応を検証して表示 |

`tracknet`・`meiji`・`chat_annotation` は3つの出自であり、3つの学習データ形式ではありません。
JPEG形式では生成後のstore利用に元動画・元画像は不要です。
`v3`は旧JPEG clipと、新規のprepared MP4を保持するclipを混在できます。
動画はsnapshot内の`media/`へ不変hardlinkし、外部の元動画pathには依存しません。
publication時に全native frameを従来と同じQ90 JPEGへ変換して、frameごとのhash・長さだけを保存します。
学習では要求された32枚をPTS照合して再生成し、hash一致後に従来のOpenCV/nvJPEGへ渡します。
追加resize・frame削減・JPEG品質変更はありません。codec環境が違う場合も黙って代用せず停止します。
永続JPEG cacheは作らず、workerあたり一窓とdecoder bufferまでを保持します。
既存の`offset/length`はvideo clipでは仮想JPEG列上の位置であり、MP4のbyte rangeではありません。
物理payloadの場所は`BallFrameStore.image_path()`、video窓は`read_video_window()`を使用します。
`v3`の教師maskは`v2`と同じで、旧clipの画素・教師・splitは保持します。

## active_ballの学習reader

公開・状態別教師・重み・group/split・修正の方針の正本は
[元動画v3の公開方針](../generate_dataset/frame_store/publishing/README.md#元動画v3の公開方針)です。
`BallFrameStore`はv4の元v3・native座標・PTSと明示maskの一致を検証し、
`video_annotations[clip_id]`から状態・遮蔽・由来・support・notesを再読込できます。
`estimated`は独立した点種として保持します。

```python
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import load_snapshot
from src.tasks.ball_detection.data.windows.active_ball import ActiveBallWindowDataset, collate_active_ball_windows
from torch.utils.data import DataLoader

snapshot, store = load_snapshot(layout, publication_receipt.snapshot)
dataset = ActiveBallWindowDataset(store, split="train", window_length=32, stride=16)
sample = dataset[0]
# 同じnative解像度の窓をbatchにする。nullを含む由来も保持する専用collateを使う。
loader = DataLoader(dataset, batch_size=1, collate_fn=collate_active_ball_windows)
```

このreaderは全native frameを覆う窓と末尾窓を作り、短すぎるclipや被覆に隙間のあるstrideを拒否します。
プレイ密度・観測点数で推定教師を落とす選別は行いません。batchの選択・モデル選択・学習起動は呼出側の責務です。
RGBは初回のarchive hash検証後に読み、使用中のファイル変更も検出します。

| 出力 | 契約 |
|---|---|
| `rgb` / `source_size` | native uint8 `(T,3,H,W)` RGB / native `(width,height)`。resizeしない |
| `xy` / `uv` | native pixel座標 / `(width−1,height−1)`端点正規化。1pixelの軸は0。無効行は0とmaskで表す |
| `position_valid` / `heatmap_valid` | 保存した位置教師mask / 正例またはactive_ball負例mask。unknownと座標なし遮蔽は両方false |
| `position_weight` / `heatmap_weight` | 対応する有効教師は1、無効は0。観測と全点種の推定は同重み |
| `direct_observation` / `estimated` / `active_ball_negative` | v3共通usageの独立mask。推定を観測へ偽装しない |
| `state` / `occlusion` / `position_provenance` | 元v3の状態・遮蔽・点種とsupport |
| `frame_indices` / `pts` / `time_base` / `native_fps` | 元動画のframe番号・実整数PTSと正確な有理数時刻係数。`timestamps`は窓先頭からの秒 |
| `target_semantics` / `recording_id` / `group_id` / `camera_id` / `split` | `active_ball`と公開時に固定した元収録・camera・分割 |

従来の`BallStoreDataset`、observed-only supervision、旧凍結window preparationはv4を明示的に拒否します。
逆に専用readerも旧v1〜v3や異なるtarget semanticsを拒否します。これによりプレー球の不在を
全ボール不在の負例へ無条件に取り込みません。既存学習CLIへの新しいdata選択や旧データ移行の実行は含みません。

## 既存sourceと教師

| source | store生成の入力 | clipとsplitの単位 | レビュー時の注意 |
|---|---|---|---|
| TrackNet | 配布画像列と`Label.csv` | clipは`Clip*`、splitはgame | 配布releaseに記録されないfpsは生成設定で明示。既存shardを使う閲覧と、原本を要する再生成を区別 |
| Meiji | multi-camera動画と保存された外部ball注釈 | clipは1カメラの連続列、splitはvideo | camera IDを保持。3カメラのclip数と元video数は異なる |
| Chat annotation | 保存済みball annotationとprepared clip動画 | clipはprepared clip、splitは元YouTube動画 | 注釈の状態と所有範囲を保存。`is_target=false`の文脈区間も注釈対象に含み、教師の採点可否とは別 |

入力位置・split設定・新規生成先は[生成設定](../configs/generate_dataset.yaml)、
正規化された注釈schemaと座標系は[store.py](store.py)、各sourceの変換は
[生成reader](../generate_dataset/frame_store/sources)を参照してください。
保存画像が縮小される場合、座標も同じwidth-ratioで変換されます。画面の位置は保存画像pxです。

レビュー画面の「教師の正例」「教師の負例」「教師対象外」「未レビュー」は、
[共通supervision実装](supervision.py)のobserved-only方針から求める互いに重複しないframe区分です。
point_kindの件数はinstance数です。複数球を含むframeでは、観測点があっても
他の位置不明・推定instanceによりframe全体が採点対象外になる場合があります。
source名や`observed`というkindだけでは、注釈者や注釈品質が独立に保証されたとは判断できません。

`v2`は採用時の教師maskを[teacher_masks.py](teacher_masks.py)の2列で保存します。
`teacher_heatmap`は0=対象外、1=負例、2=正例、`teacher_coordinate`は0=対象外、1=単一観測球です。
両列255は既存clipの解釈を維持する明示的なlegacyモードで、clip内ではモードを混在できません。
新規採用clipでは、未レビュー・文脈・未解決・補間・遮蔽推定・採用を妨げる注記を教師へ昇格させません。
レビュー済みの空frameでも、不在が確定しなければヒートマップの負例にはなりません。
学習時のpoint-kind方針はmaskをさらに絞れますが、除外済みの教師を復活させられません。

`annotated`・元の座標・point_kindはレビュー事実として保持し、maskと混同しません。
ヒートマップ、座標窓、candidate/holdout評価、レビュー画面の教師区分はmaskを適用します。
座標窓のプレイ存在証拠は元のレビュー事実から計算し、座標教師への採用とは区別します。
clipの意味的hashには明示maskも含める一方、legacyモードの既存clipは元のhashを保ちます。
`v2`でmask欠落・不正・注釈との矛盾があれば読込みを失敗させます。
既存生成CLIは引き続き`v1`を出力し、ここで既存storeを書き換えることはありません。

同じversionに追記されたstoreは、version名だけでは同一内容と判断できません。
`metadata.json`の`append_history`と`index.npz`を確認します。
派生pose storeはmanifestの元snapshotへ固定されており、元snapshot後に追加されたclipへ
poseを補完しません。採用済み・要確認・レビュー待ち・未生成・対象外・生成エラーを区別します。
画面は保存結果を読むだけで、生成やレビューcampaignを再開しません。

旧Web/static/temporal・staged・SSL収集の学習入力は、
[`515370dfc9a09012d8201e497a89b884d2b07c9e` の整理](https://github.com/Motoki0705/tennis-lab/commit/515370dfc9a09012d8201e497a89b884d2b07c9e)で廃止されています。
旧版や旧原本が手元にないことを、現行レビューUIの不足データとは扱いません。
生成設定の既定versionと学習設定の既定versionは用途が異なります。
新規生成・split変更には新しいversionを明示し、既存データを閲覧だけのために作り直さないでください。
