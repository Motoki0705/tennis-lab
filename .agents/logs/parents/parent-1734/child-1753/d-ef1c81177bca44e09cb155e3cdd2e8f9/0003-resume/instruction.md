# PR #2226 正式レビューのmajor指摘への対応（親 #1734 から #1753 へ）

正式レビュー（gpt-6.2-astra / xhigh、HEAD 698129d6e）で、major指摘が5件出た。親はどれも妥当と判断した。5件とも直してください。人間の指示で、再レビューは行わない。親が修正とCIを確認して、そのままmergeする。各修正は、根拠（テスト・確認の結果）とともにPR本文に記録すること。

## 1. 部分キャッシュから再開するとViTPoseとCLIPが同時に常駐する

実行検証。標準定義と同じ順序・依存関係の2カメラを対象HEADの `InferenceRuntime` とCPUの代替モデルで実行し、cam0の人物検出だけをキャッシュヒットにしました。cam1の検出より先にcam0のposeとCLIPが実行され、cam0のCLIP実行中にはViTPoseとCLIP、続くcam1の検出中にはDINOを含む3モデルをruntimeが保持しました。cam1に残るpose需要がViTPoseの解放を妨げます。#1753の別pass・VRAM要件を満たすには、部分ヒット時もカメラをまたいでモデルの実行順を揃え、この再開条件で同時常駐を検査する必要があります。

## 2. loadで取り込んだ特徴はモデルや精度が異なっても追跡に使われる

実行検証と静的推論。対象HEADの `ExecutionPlan.decide()` に、同じ検出参照を持つimport由来のFP32 poseと、別checkpoint・FP16を宣言したload nodeを渡すと、判定は `load` でした。新しい特徴nodeにはimport時のモデル照合がなく、`PersonPoseFeaturesOutput` にも生成モデルのidentityがありません。CLIP側には `encoder_identity` がありますが、`PersonTrackingModule.process()` が追跡前に照合するのはencoder名だけです。同名で重みの異なる特徴もStrongSORTに入り、選別で不一致を検出する時点では追跡IDが確定しています。特徴を採用する境界で期待するModelSpecとの一致を確認し、確認できない入力は追跡前に拒否する必要があります。

## 3. 特徴nodeを差し替えるとcrop設定がキャッシュキーから抜ける

静的推論。`standard_definition(overrides=...)` は差し替えたnodeの設定を `component_override_identity()` から取得します。新しい2つの特徴moduleは `cache_identity()` を持たず、`FeatureConfig` はModelSpecにも含まれません。このため、同じ重み・入力のままposeの `bbox_enlarge` を変えても、CLIPの `min_appearance_height_px` を変えても、それぞれ変更前と同じnode identity・完成markerになります。前者は切り出す領域、後者は `appearance_valid` になるrowを変えます。各moduleが実際に使う前処理設定をidentityへ含め、設定変更後に旧成果物がヒットしないことを確認してください。

## 4. ViTPoseの精度だけを変えても全カメラのCLIPを再計算する

静的推論。`definition.py` は `person_appearance_features` の入力にpose成果物を束縛しますが、`person_features.py` が標準CLIPのcrop関数へ渡すposeは常に `None` です。ViTPoseをFP32からFP16へ変えるだけで上流参照が変わり、画像・box・CLIP重みがすべて同じでも、既存の外観特徴はexecuteで再計算され、明示loadでは依存不一致になります。外観特徴だけを要求した場合もViTPoseが必要です。モデルの実行順と成果物の入力依存を分け、poseだけを変更したときに保存済みCLIPを再利用できるようにする必要があります。

## 5. 新しいpose成果物を既存のレビューUIとBlender出力が読めない

静的推論。`PoseEstimationModule` の出力は `person_poses` v2になりましたが、`src/submodules/visualization/review/store.py:31` はv1だけを許可し、`src/tennis_scene/visualization/blender/export/placement.py:407` もv1との完全一致を要求します。新しい標準storeを渡すと、GVHMR／poseレビューUIは起動時に、Blenderの配置出力・ランキングは入力読込時に停止します。`tests/benchmarks/ball_detection/detection_holdout.py:77` にも同じ制限があります。履歴v1の閲覧を保ちつつv2を受け付け、現行producerの版を使って各readerを検証してください。

## 進め方
- 1: 一部だけキャッシュにヒットした状態から再開しても、カメラをまたいでモデルの実行順を揃え、同時に載るモデルを制限する。この再開条件で、モデルの読み込み・解放の順序を検査するテストを追加する。
- 2: 特徴を採用する境界（loadとimport）で、期待するModelSpec（重みのSHAと精度を含む）との一致を確認する。確認できない入力は、追跡の前に拒否する。CLIPの照合は、encoder名だけでなく完全なidentityで行う。
- 3: 各特徴moduleが実際に使う前処理の設定（bbox_enlarge、min_appearance_height_pxなど）を、cache identityに含める。設定を変えたあとに古い成果物がヒットしないことを、テストで確かめる。
- 4: 外観特徴（CLIP）の成果物の入力依存から、使っていないposeを外す。モデルの実行順の制御は、成果物の依存とは別の仕組みで行う。poseの精度だけを変えたとき、保存済みのCLIPを再利用できることをテストで確かめる。
- 5: レビューUI・Blender・ベンチマークの読み込み先で、person_poses v2を明示的に受け付ける。v1の閲覧も残す。現行のproducerの版を使って、各読み込み先をテストする。
- 注意: 並行してPR #2225（#1754）も、同じ読み込み先（review/store.py、blender placement.py）でbody_parameters v2への対応を入れている。版を扱う仕組みは、あとで両方を統合しやすい形にすること（たとえば、成果物ごとに対応する版の集合を宣言する）。
- 同じ種類の問題（版の固定、identityの漏れ、使っていない依存）がほかにないか、横断的に確かめ、結果の表をPR本文に載せる。
- tests/unit/tennis_scene 全体と関連テスト、pre-commitを実行し、pushしてPR CIで全ジョブの成功を見届ける。GPUは使わない。
- CUDAのエラーやハングが起きたら、再試行せずに止めて報告すること。最後に、HEADのSHA・CIのURL・各指摘への対応を簡潔に報告すること。
