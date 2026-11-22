# PR #2225 正式レビューのmajor指摘への対応（親 #1734 から #1754 へ）

正式レビュー（gpt-6.2-astra / xhigh、HEAD 8efdfe1fa）で、major指摘が3件出た。親はどれも妥当と判断した。3件とも直してください。人間の指示で、再レビューは行わない。親が修正とCIを確認して、そのままmergeする。そのため、各修正は根拠（テスト・確認の結果）とともにPR本文に記録すること。

## 1. 一部のclipだけに保存特徴があると、GVHMRの常駐中にHMR2を読み込む

CPU実行検証。実際の PlanExecutor と InferenceRuntime に、PRと同じ依存関係のfakeモデルとメモリ内storeを接続しました。clip AのHMR2特徴まで保存し、A・Bを既定順で再開すると、イベントは「GVHMR読込 → HMR2読込 → HMR2解放 → GVHMR解放」でした。Aがキャッシュにヒットすると、既定schedulerはBの特徴抽出より先にAのGVHMRを選び、BにもGVHMRの需要が残るため解放しません。各モデルの読込・解放は各1回でも、GPUでは両方の重みが同時に載ります。全clipのHMR2処理を終えてからGVHMRを開始する順序を保証し、既存の複数clipテストにこの混在条件とイベント順の検査を追加してください。

## 2. 公開だけを再開すると、分離前のsceneに現行の完了マーカーを付けられる

静的な推論。`direct_annotation.resume=publication` は `standard_definition` を通らず、`saved_scene()` が保存済みarchiveを返します。ここではreceiptと `scene_assembly` の参照、精度、ファイルのハッシュを照合しますが、GVHMRが `body_parameters` v2で、保存済み `hmr2_features` に依存しているかを確認しません。分離前にscene生成まで完了し、datasetの完了マーカーを書く前に中断した場合、`generate_dataset` はv1に由来するsceneを読み、現行の `publication_identity_sha256` で `annotation.json` を発行できます。現行のマーカーを発行する前に、公開対象のsceneから身体成果物の版と依存参照をたどって照合してください。[公開再開の分岐](https://github.com/Motoki0705/tennis-lab/blob/8efdfe1fa0189b611e49a8f8a341dc0c0ec3c905/src/tennis_scene/pipeline/orchestrator.py#L220)・[マーカーの発行](https://github.com/Motoki0705/tennis-lab/blob/8efdfe1fa0189b611e49a8f8a341dc0c0ec3c905/src/tennis_scene/pipeline/dataset_publication.py#L199)

## 3. 新しい身体成果物をPoseレビューUIとBlenderの配置exportが読み込めない

静的な推論。GVHMRの出力は `body_parameters` v2になりましたが、`PoseReviewStore` はv1だけを許可するため、新しく生成したstoreを開くと `Unsupported gvhmr: body_parameters v2` で停止します。Blenderの `read_placement_inputs()` もv1を固定指定しており、配置bundleのexportと配置候補のrankingが両方停止します。両方の読込先でv2を明示的に扱い、現行のGVHMR出力を渡すテストで確認してください。[Poseレビューの対応版](https://github.com/Motoki0705/tennis-lab/blob/8efdfe1fa0189b611e49a8f8a341dc0c0ec3c905/src/submodules/visualization/review/store.py#L28)・[Blenderの読込条件](https://github.com/Motoki0705/tennis-lab/blob/8efdfe1fa0189b611e49a8f8a341dc0c0ec3c905/src/tennis_scene/visualization/blender/export/placement.py#L383)

## 進め方
- 1: 一部のclipだけに保存特徴がある状態から再開しても、HMR2とGVHMRが同時にGPUに載らないようにする。全clipのHMR2を終えてからGVHMRを始める順序を、schedulerの依存か段階で保証する。混在条件と、モデルの読み込み・解放の順序を検査するテストを追加する。
- 2: 公開だけを再開するとき、公開するsceneの身体成果物の版（body_parameters v2）と、hmr2_featuresへの依存を照合する。一致しなければ明示的に失敗させる。テストで固定する。
- 3: PoseReviewStoreとBlenderの配置読込で、v2を明示的に扱う。v1を黙って読み続けるのではなく、対応する版を明示する。現行のGVHMR出力を渡すテストで確認する。
- 同じ種類の問題がほかの読み込み先（body_parametersを読む箇所すべて）にないか、grepで横断的に確かめ、結果の表をPR本文に載せる。
- tests/unit/tennis_scene 全体と関連テスト、pre-commitを実行し、pushしてPR CIで全ジョブの成功を見届ける。GPUは使わない。
- CUDAのエラーやハングが起きたら、再試行せずに止めて報告すること。最後に、HEADのSHA・CIのURL・各指摘への対応を簡潔に報告すること。
