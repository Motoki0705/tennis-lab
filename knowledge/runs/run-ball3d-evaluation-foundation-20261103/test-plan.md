# #1556 実装前のテスト設計

親 #1554 の2026-11-03合意と #1556「基盤のテスト設計（実装前）」を適用する。
実装は既定Codexの共通ランナーで行い、推論・GPU・再注釈は行わない。

| 契約 | 検出したい不具合・期待値の根拠 | 層・既存node | コスト上限 |
|---|---|---|---|
| 固定入力のcamera/frame・px・fps・mask・出自を保持し、破損を拒否 | 非対称camera ID/焦点距離/欠測frameを用い、camera入替、frame shift、mask改変、checksum不一致、注釈originの混入を故障注入する。descriptorとsourceの独立記録が期待値 | unitを基本とし、component codecからbundleへの保存読込だけintegration。既存 `tests/unit/tennis_scene/pipeline/test_component_store.py::test_source_mismatch_cannot_reuse_another_clip` と `tests/unit/tennis_scene/pipeline/components/test_ball_evidence.py::test_artifact_roundtrip_is_exact_and_load_only_does_not_infer` を実行 | 新規30 case以内、parametrize 50以下、unit各5秒/setup1秒以内を目標。collect-onlyとdurations全件を記録 |
| 方式へ注釈・reference 3D・除外cameraを渡さない | spy方式で実際のrunner入力を確認し、別annotationで採点しても方式結果が同一。可視性maskはdetector由来のみ | unit。runnerと結果の保存読込の接続は小さなintegration | 同上。動画・checkpoint不要 |
| LOVOで保持viewだけを採点し前後半と欠測を分離 | 非対称3camera、手で指定した3Dと投影座標から独立に5/13/25px誤差を作る。期待値を手計算し、非保持viewと未知点の巨大な座標が集計へ入らないことを確認 | unit。oracleは独立に指定した画素観測から既知3Dへ戻ることを確認 | 各数frame、全1010frame fixtureを量産しない |
| baselineを実storeから再現 | 保存されたdecode/gate・LOESS設定をCPUで再実行し、単一観測・ball_points・3D位置・有効/使用view maskを保存値と照合。元codeを新codeと偽装しない | 実clipはCI fixtureにせず、knowledgeの実行scriptと数値receiptで別途検証 | 実所要時間を別記。unitの5秒基準と混同しない |
| 表示が欠測を跨がずN方式を扱う | 非対称な欠測を持つ軌跡の線分、全view/LOVOの軸、HTMLのlocal参照を確認。実動画は全frame decodeと代表画像の目視 | 純粋な線分・集計はunit、小動画への出力だけintegration | 最小動画、setup1秒を目標。実clipの動画検査は別計測 |

既存テストの期待値や実装の式をコピーして正解を作らない。合成fixtureは契約検査専用で、
実clipの正解ではない。detector予測をGTへ流用しない。

2026-11-03 再開時の追加設計（親の合意 revision 6）:

- 正解JSONの検証は #1668 のreaderへ一本化し、`UserReference.select` の採点maskとkindを変換する層だけを残す。旧 `test_annotations.py::test_sparse_native_pixel_labels_preserve_camera_frame_and_unreviewed_gaps` と `test_annotation_source_and_review_contract_rejects_untrusted_points`（5 case）はGPT専用schemaの廃止に伴い置換する。source/schema/hashの拒否は `tests/unit/tennis_scene/ground_truth/test_reader.py` に移管済みの契約を再利用する。追加するadapterテストは同readerの非対称fixtureで座標・推定・除外・kind・元イベント/区間の保持を確認する。
- 採点の手計算fixtureを拡張し、同じ5/13/25 pxのうち推定点を13 pxに固定する。全点と直接観測・推定の分布、分母と欠測coverage、oracleの推定利用をunitで確認する。カメラ/frame不一致をadapter境界で拒否し、runnerの入力からreferenceと採点maskが隔離されることを既存spyテストで確認する。
- イベントは物理イベント数を推定せず、camera別の元25行を保存する。kindごとに元行の±3frame窓の和集合で残差を1回だけ採点し、行数を物理イベント数と表示しない。同kindの0/1frame差と窓の重複を手書きfixtureで検査する。
- 動画・図は直接観測と推定に別の印を使う。小動画のintegrationを2方式で実行し、区間別表・分布・時系列・欠測・HTML埋め込みを検査する。全viewとLOVOの結果は実clipで保存・再読込し、全動画frameを逐次decodeする。
- 新規/変更分は合計40 case以内を目標とし、既存reader50 caseと指定の再利用2 nodeを含め `-n 0`（最大2）でcollectと全durationを記録する。codec・描画は初回importを含む実測を分け、メモリの重い処理は並列にしない。

結果の保存復元は既存integration nodeで、非対称な更新intrinsic・共分散・秒単位offsetも保存して照合する。LOESSなしの座標と採用maskの契約を専用unitで確認する。

## PR #1669 レビュー修正の追加設計（実装前）

- pr-1669-1: source courtのcamera別raw KP14/score/valid・homography採用subsetと実観測frameを保持する。broadcast済みfitted14をraw観測と偽装せず、LOVOで除外viewのcourtデータが入らないことを、camera毎に異なる画素値・採用点を持つfixtureで検証する。camera/frame/schema/欠落をunitで拒否し、実descriptorからの接続を既存bundle integrationで検証する。元storeにLINEの生観測はなく要約のみであるため、この欠落の扱いは親へ確認中。観測を捏造しない。
- pr-1669-2: 採点APIを検証済みFixedInputBundleへ束縛し、全runのdigest・camera/frame軸と、正解のclip/camera/frame/fps/nativeサイズ/動画hashを必須照合する。既存の手計算scoring fixtureとannotation adapter nodeを拡張し、同camera・同Tの別digest、別clip、短い結果、別video hashの拒否をunitで検証する。plot/cropの残差抽出も同じ境界を使う。
- pr-1669-3: HTML主表とPNGは同じ表データ生成を使い、LOVOのcount/label_count/coverageを誤差の隣、全view有効率を別見出しにする。レビュー実例の825/1462=56.43%と全view83.17%を手書きfixtureで区別する。全方式が採点できた(camera,frame)集合の共通部分での値も出し、欠測を0誤差として補完しない。片方のLOVOだけ1点欠く非対称fixtureで共通集合を独立に手計算する。
- 追加は20case以内、parametrize50以下、unit各5秒/setup1秒以内を目標。既存80件と追加分を-n 0でcollect/durations全件計測する。実動画の再生成・全frame decode・目視とHTML参照切れは新しいfoundation_v2で別に測定する。foundation_v1と固定入力v1を変更しない。
