# 凍結学習窓・共通区間と全フレーム教師

## 全フレーム教師の可変長窓

`all_frames.py`は、clip内のreference-only frameと大きいPTS gapを境界として、
最大T以上の連続区間をすべて採る。プレイ密度・位置教師率では絞らない。
`AllFrameWindowConfig`がパラメータの正本で、実行値を新schema
`ball_all_frame_windows.v1`の`config`へ保存する。既定は最大T=pack長128、
pack開始stride16、frame_step=1、gap factor=2.5。Tはpack長を割り切る2以上の偶数とする。

- 学習は区間内のpack長の列をstrideで置き、未登録の末尾開始位置を1回足す。
  observedまたは確認済み不在が1枚以上あるpackだけを採用し、T枚ずつに分割する。
  128枚なら32×4・64×2・128×1で同じframe出現列になる。全maskの子窓も残す。
- 評価は学習packと独立に、共通区間の全frameをstride T/2＋末尾補完の窓で覆う。
  `(abs(2*frame-(2*start+T-1)), start)`が最小の窓がframeを所有する。
  窓ごとの半開`owner_range`へ割当を凍結し、未知frameも含めて一度ずつ所有する。
- 教師は単一observedを位置・heatmap正例、`reviewed_empty`と単一`out_of_frame`を
  全0のheatmap負例とする。推定・未解決・複数球・未レビューはmaskのまま。
  Chat空ラベルは`hybrid.reviewed_empty_evidence`で原注釈のhash・frame対応・notesを照合する。
  採用済みteacher maskを持つstoreではその除外も維持する。不在専用集合は作らない。

### 凍結入口

既存snapshotから、新しい絶対pathへ生成する（学習は開始しない）:

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.prepare_data --mode all_frames \
  --ball-store /absolute/immutable-ball-snapshot \
  --output /absolute/new-all-frame-data --window-lengths 32 64 128 \
  --max-window-length 128 --pack-length 128 --pack-stride 16 --timestamp-gap-factor 2.5
```

`all_frame_preparation.prepare_all_frame_training`はsnapshotを**読み取り専用で参照**する。
RGBのコピー・hardlink・既存snapshotの変更は行わない。snapshotは生成後も同じpathで不変に保つ。
`--rgb-manifest /absolute/mdd_only_windows.json`を明示すると、同一snapshotのindex/metadataと
clip semantic identityを照合して既存のRGB hashを再利用する。未収録clipは実ファイルをhashする。
datasetはどちらの場合も初回読込時に画像ファイルの内容を検証する。
配布用storeへの移動・clip indexの変更は別工程で、clip_idによる明示的な対応付けが必要。
原Chat注釈は`store.annotation_path(clip)`で解決し、元snapshotの絶対pathと配布storeの
store相対pathの両方で同じhash・frame対応・notes検査を行う。

RGBを未配置の配布stageでローカル検証する場合だけ、`--distribution-source-store`へ
固定された元snapshotの絶対pathを明示する。配布manifestの元index/metadataとclip対応を検証し、
元RGBを読み取り専用で参照する。自動探索・複製・hardlinkは行わない。
planの`payload_mode`は`external_source_view`となり、出力manifestの通常dataset読込には
配布先への全payload配置が必要である。配置済みstoreではこのoptionを省き、
`self_contained`として生成・読込する。`--rgb-manifest`とは併用しない。

出力はTごとの`train_T<T>.json`（pack単位）、`evaluation_T<T>.json`（所有範囲付き窓）、
およびpath・hash・コード版を持つ`experiments.json`。
すべてのmanifestを検証し、入力ファイルのinode/size/mtime/ctimeが変わっていないことを確認してから
planを公開し、`PREPARATION_INCOMPLETE`を取り除く。失敗した出力は読込を拒否する。
常設実装はsnapshotと注釈から再生成し、knowledgeの調査ファイルを実行時入力にしない。

各manifestの`summaries[split]`には全共通区間の`common_intervals_sha256`、全frameの`target_identity_sha256`、
frame key末尾に教師codeを加えた`teacher_identity_sha256`、既知教師だけの`scored_frame_sha256`、
pack列の`training_packs_sha256`と件数を保存する。これらはT・学習/評価で一致する。
frame keyはclip_id順・frame順のcompact JSONL `[clip_id, frame_index, 1, exact_native_fps]`。
評価の`owner_identity_sha256`はTごとの所有開始列を別に記録する。

### datasetとpack境界

`FrozenBallWindowDataset`は旧`ball_play_windows.v1`・coordinate v1・hybrid v1を32枚のまま読み、
新schemaでは宣言したT枚を読む。schemaの付替え・混在、区間・教師・pack・owner・hashの不一致は
画像読込前に拒否する。新schemaでは原注釈の検証済みkindと来歴を凍結するため、
dataset読込時にChat原注釈を開き直さない。RGBはnative解像度・uint8、時刻は実PTSからの秒とする。

既存の`uv`・`position_valid`・`heatmap_valid`・`frame_indices`に加えて、新schemaは次を返す。

| field | 意味 |
|---|---|
| `label_kind` | T枚のuint8。0=masked、1=observed、2=reviewed_empty、3=out_of_frame |
| `pts` / `time_base` / `native_fps` | nativeの整数PTS、秒への分数係数、正確なFPS |
| `pack_index` / `pack_id` / `pack_start` | split内のpack番号、clip/start/pack長によるID、開始frame。評価では`None` |
| `child_index` / `children_per_pack` | pack内の子窓番号と個数。重なるpack間の同一子窓も別出現として保持 |
| `owner_mask` / `score_mask` | 評価窓の所有mask / 所有かつheatmap教師あり。位置採点はさらに`position_valid`で選ぶ。学習では両方false |
| `interval_start` / `interval_stop` | 所属する半開共通区間 |
| `edge_distance` / `edge_bin` | 区間端までの最小frame数 / 0、1–7、8–15、16–31、32–63、64+のcode 0〜5 |
| `edge_distance_seconds` | 実PTSでの最小区間端距離。窓内の左右contextは`frame_indices-start`とTから求める |

`FrozenBallPackDataset(train_manifest, split="train", jpeg_decoder=...)`の1要素は全子窓のlist。
`collate_pack_samples`は完全なpackを並べて通常の窓batchへまとめ、`pack_offsets`で
各抽出出現の子窓境界を返す。同じpackを復元抽出しても出現を結合しない。
組列のRNG・抽出位置、pack全体での拡張、更新全体の損失正規化は学習loopの責務。
拡張を使う場合は分割前の共通packへ同じ変換を適用し、MDDは分割後の各RGB窓から再計算する。
評価には`evaluation_T<T>.json`を窓datasetで読み、`score_mask`を使う。

## 既存のプレイ区間レビュー

公開済みの累積snapshotと固定validation/testを追加学習へ接続するAPIは
[learning/](learning/README.md)を参照。新規clipの座標教師が不足する場合の停止と被覆記録もここが担当する。

`intervals.py`は注釈からプレイ区間を提案する純粋関数。
推論時のプレイ検出器ではなく、固定済み学習データの選択にだけ使う。
設定値の正本は`PlayIntervalConfig`、実行値は生成manifestへ保存する。

1. レビュー済み・target frame・単一球のobserved/interpolated/occlusion_estimated/unresolvedを存在の証拠にする。
2. 前後に証拠がある短い欠損だけを実PTS秒で連結する。区間端を延長せず、reference-only frameと時刻の大きな飛びを跨がない。
3. 実frame数と存在証拠率を満たす連続区間をプレイ候補とし、残りを非プレイ候補とする。
4. 各候補内で固定長の窓を切り出す。位置教師が十分ある窓だけを学習に採用する。

`segment_break`はhit/bounce/cut/play boundaryを混在して表すため、一律に切断しない。
複数球・未レビュー・out_of_frameは存在の正証拠にしないが、短い内部欠損として
連結され得る。連結は区間選択だけを変え、座標教師や存在ラベルを書き換えない。

**プレイ候補と学習窓の被覆は別**。unresolved主体のプレイ区間は位置教師不足で
学習対象にならない場合がある。非プレイ候補は確定GTではなく、短いラリー・長い遮蔽・
画面外への移動を誤って除外し得る。逆に静止球や打球準備を含む可能性もある。

レビューは[データセットWebUI](../../visualization/README.md#プレイ区間の候補)で行う。
選択したclipの注釈からCPUでその場で計算し、画像・GTと同じframe列に区間を表示する。
ファイル出力・モデル推論・学習は行わない。通常storeの候補とpose承認済みsubsetは
別のdatasetとして表示し、後者はpose manifestが指定したball snapshotを検証して読む。
カタログ更新で承認状況を再読込するため、UI表示自体は学習対象の凍結や承認ではない。

レビュー後に学習対象を固定するAPIは
`pose_manifest.build_play_manifest(pose_directory, PlayIntervalConfig(...))`。
返り値をJSONとして実験出力先に保存すると、approved subset、ball snapshot、
pose artifact、JPEG shardのhash、学習窓と全除外範囲を固定できる。
この保存処理はWebUIの閲覧とは分離し、生成物をソースリポジトリへ追加しない。

## FPSと32frame窓

上のAPIと既存WebUIはnative-FPSの候補を扱う。
`temporal.py`はプレイ区間内で元FPS/1/2/1/4（`frame_step=1/2/4`）の32枚を選び、
選んだframeの証拠率・observed教師数を改めて判定する。
窓のframe列は`start + frame_step * arange(32)`、開始strideの単位は**sampled frame**。
`window_stride=16`なら元frameでの開始間隔は16/32/64枚となり、区間末尾を覆う窓も追加する。
プレイ区間はnative PTSから決め、窓はreference-only frameと大きいPTS gapを跨がない。
時刻は選んだframeの実PTS秒を使い、一定FPSの時刻を捏造しない。

`dataset.py`は固定manifestからRGB・任意のpose・observed座標と教師maskを同じ32枚で返す。
RGBを間引いてからモデル内でMDDを再計算する。native-FPSのMDDを間引かない。
5ch構成でもdatasetは同じRGBを渡し、拡張後にモデルがRGBとMDDを結合する。
入力契約は [models/mdd](../../models/mdd/README.md) を参照。
crop/resize、時間の反復、欠損位置の補完はしない。
座標lossは`training/coordinates/objective.py`が担当する。

## 学習manifestの凍結

`manifest.py`は`mdd_pose`をpose承認済みclipから、`mdd_only`をposeの有無によらず全clipから選ぶ。
`mdd_only`にはclip全体の存在率による足切りを設けず、プレイ区間と窓ごとの条件を使う。
Common評価はpose参照とGT・split・timeline・RGBのidentityが一致するsubset、Full評価は全validation。

レビュー後、次の公開入口で新しい絶対pathへ凍結する（学習は開始しない）:

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.prepare_data \
  --ball-store /absolute/ball-store --poses /absolute/approved-poses \
  --model-config /absolute/configs/model/mdd_pose.yaml \
  --output /absolute/new-prepared-data --frame-steps 1 2 4 --window-stride 16
```

`preparation.py`はindex/metadataとpose artifactをコピーし、RGB shardを`ball_snapshot/`へhardlinkする。
`mdd_pose_windows.json`・`mdd_only_windows.json`に窓・除外clip・入力hash・評価subsetを保存し、
`models/`に比較設定を出力する。各split/FPSの窓とCommon評価窓を検証した後に
`experiments.json`を公開し、`PREPARATION_INCOMPLETE`を取り除く。完了したplanのmanifest/model pathを
[学習入口](../../training/README.md)へ渡す。凍結出力はソースリポジトリへ追加しない。

## snapshotを共有するrunの運用

同じ元storeのshardを共有するsnapshotを学習・評価runが読んでいる間は、
`prepare_data`によるsnapshotの追加と、既存snapshotの削除を行わない。
hardlinkの作成・削除は共有inodeのctimeを変えるため、RGB内容が同じでも、
検証済みdatasetの次回読込や`FileRangeReader`の範囲読込は変更を検出して停止する。
必要なsnapshotをrun開始前にまとめて準備し、削除は共有する全runの終了後に行う。
この検査はmtimeを戻した内容変更も検出するため、ctimeの照合を省略しない。
