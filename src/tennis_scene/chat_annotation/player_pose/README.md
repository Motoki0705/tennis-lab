# ボール検出用の選手姿勢・IDレビュー

ball frame storeを入力に、全画面DINO → ViTPose-H/CLIP-ReID →
既存StrongSORT++＋pose/AFLinkを実行し、rawトラックをGPTの視覚レビューへ渡す。
コート推論・校正・人物選別は呼ばない。匿名選手IDの決定はGPTの区間対応表で行う。
移動カメラでも追跡方式は既存のまま使い、分裂・切替をレビューで扱う。

存在率はレビュー済み対象frameのうち、observed/interpolated/occlusion_estimated/
unresolvedの球が1つ以上あるframeの割合。out_of_frameは数えない。
参考frameは分母から外す。未レビューframeがあるclipは保持し、その理由を記録する。
閾値以下はpose生成をスキップする。球GTやRGBストアを削除しない。

入口は .venv/bin/python -m src.tennis_scene.chat_annotation.player_pose。
init --campaign <絶対path> --config <JSON> でplanと入力・重みhashを固定する。
orchestrateをCPUで起動すると、clipごとに共有training queueへallジョブを登録する。
生成済みclipから順にCodex CLIを並行起動し、GPTの判断を検証して保存する。
再開時には所有しているqueue jobを監視し、重複登録しない。
statusは対象数・生成・レビュー状況をJSONで返す。

configはstore/dataset/project_root、presence_threshold、assets、
chunk_frames、generation_attempts、cuda_memory_fraction、codex_binary/codex_home、
model/effort、review_parallel/review_attempts/review_timeout_secondsを明示する。
assetsにはdino、dino_repository、vitpose、clip_reid、aflink、dino_extensionを指定する。
GPU環境のPYTHONPATHには指定したDINO拡張libと実装worktreeを含める。

検出・姿勢特徴はframe chunkごとに原子的に保存する。新しいCUDAプロセスで再試行し、
完了chunkをhash照合して再利用する。3clip連続失敗では停止し、原因を隠さない。
raw ID数を切り捨てず、選手人数も固定しない。GSI補間はpose観測へ変換しない。

レビュー指示の正本はWORKER.md。全frameのID overlay一覧と人物crop一覧を作る。
JSONにはraw ID・半開frame区間・匿名player ID・役割・根拠frameを保存する。
全raw観測の重複ない被覆、同一frameでのplayer ID衝突、duplicateの相手、
入力hashと画像一覧のcoverageを検証する。曖昧な区間はneeds_reviewとして保留する。
未検出frameにposeを創作せず、欠損maskを保持する。利用枠制限は30分待って再開する。

キャンペーンはoutputs配下のconfig/plan、clipごとのdetections/features/tracks、
evidence、review attemptとログを持つ。採用データは別のdatasetディレクトリの
manifest.json、clips/*.npz、reviews/*.jsonに保存する。RGBを複製しない。
PlayerPoseStoreはball storeのhash・frame/PTS・姿勢artifactを検証し、
未採用clipを黙って空poseにしない。スキップされたclipだけNoneを返す。

再開は同じorchestrate入口を使用する。稼働中は実装・入力・重みを変更しない。
テストはtests/unit/tennis_scene/chat_annotation/player_pose。実GPU/modelを使わない。

