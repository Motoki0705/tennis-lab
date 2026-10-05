# Ball Detection のデータセット体系

学習・評価・レビューの現行入力は、`data/ball_detection/<version>` の統一ball frame storeです。
現物のversion・件数・source/split内訳はレビューUIの「体系・内訳を見る」で確認します。
その画面は検証済みindexから集計するため、READMEへ件数を重複管理しません。

| 系列 | コード対応・保存先 | 入力と教師 | 用途・制限 |
|---|---|---|---|
| 統一ball frame store | `ball_detection_frames.v1`。`BallFrameStore` がversionを読む。通常学習の既定は `ball-mix-v2` | 1カメラの連続RGB JPEGと、point_kind付き2D ball注釈。全フレームを保存 | 学習・validation・test・データ品質レビュー。使用source・混合比・教師方針は[学習data設定](../configs/data/rgb_sequence.yaml)が正本 |
| 派生player pose store | `ball_detection_player_poses.v1`。`data/ball_detection/*/manifest.json` と保存済みcampaign artifact | ball storeのRGBからモデル推定した2D pose・人物枠・クリップ内選手/raw ID | ボールレビューの補助表示。ボール教師や独立したpose GTではない。manifestの元ball store hashとclipの対応を検証して表示 |

`tracknet`・`meiji`・`chat_annotation` は3つの出自であり、3つの学習データ形式ではありません。
生成後のstore利用に元動画・元画像は不要です。

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

同じversionに追記されたstoreは、version名だけでは同一内容と判断できません。
`metadata.json`の`append_history`と`index.npz`を確認します。
派生pose storeはmanifestの元snapshotへ固定されており、元snapshot後に追加されたclipへ
poseを補完しません。採用済み・要確認・レビュー待ち・未生成・対象外・生成エラーを区別します。
画面は保存結果を読むだけで、生成やレビューcampaignを再開しません。

旧Web/static/temporal・staged・SSL収集の学習入力は、
[`5e8e559d9` の整理](https://github.com/Motoki0705/tennis-lab/commit/5e8e559d9)で廃止されています。
旧版や旧原本が手元にないことを、現行レビューUIの不足データとは扱いません。
生成設定の既定versionと学習設定の既定versionは用途が異なります。
新規生成・split変更には新しいversionを明示し、既存データを閲覧だけのために作り直さないでください。
