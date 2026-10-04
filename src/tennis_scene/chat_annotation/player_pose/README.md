# ボール検出用の選手姿勢・IDレビュー

ball frame storeから、全画面DINO → CLIP-ReID → poseなしStrongSORT++/AFLink →
GPT画像レビュー → 選択選手の実観測だけViTPose-H → pose dataset公開の順に処理する。
コート推論・校正・自動選手選別は呼ばない。匿名選手IDはGPTの区間対応表で決定する。
通常sceneの採用profileは変更しない。生成専用profileの契約は
[person_tracking](../../../tasks/person_tracking/README.md)を参照する。

存在率はレビュー済み対象frameのうち、observed/interpolated/occlusion_estimated/
unresolvedの球が1つ以上あるframeの割合。out_of_frameは数えない。
参考frameは分母から外す。未レビューframeがあるclipは保持し、その理由を記録する。
閾値以下は生成をスキップする。球GTやRGBストアを削除しない。

入口は `.venv/bin/python -m src.tennis_scene.chat_annotation.player_pose`。
`init --campaign <絶対path> --config <JSON>` で新しいplanと入力・重み・実装hashを固定する。
`generation_mode` は `review_then_pose.v1`（新規initの既定）のみ。既存campaignの
設定やraw tracksを後から変更せず、旧campaignの実行にはその固定worktreeを使う。
`store`には固定したmetadata/indexと対応するJPEG shardsを持つsnapshotを指定する。

configにはstore/dataset/project_root/python、presence_threshold、assets、chunk_frames、
generation_attempts、cuda_memory_fraction、codex_binary/codex_home、model/effort、
review_parallel/review_attempts/review_timeout_seconds、queue_dir/queue_script/session_idを明示する。
assetsにはdino、dino_repository、vitpose、clip_reid、aflink、dino_extensionを指定する。
`pose_attempts`は任意で、省略時はgeneration_attemptsと同じ回数。
GPU環境のPYTHONPATHには指定したDINO拡張libと実装worktreeを含める。

拡張時は`reuse_dataset`で旧approved datasetを指定する。clip/media/annotation identity、
座標系・解像度・frame/PTS・画像hash・レビューとraw poseの対応を照合する。
一致するapproved poseだけを新datasetへ保存し、出自を`legacy_pose_before_review.v1`として残す。
単なるpath差し替えは行わない。追加・変更・既存clip、生成・再利用・skip件数はplanへ記録する。
旧subsetを使った実験は旧manifestを、新たな拡張実験は新manifestを指定して区別する。

`orchestrate --campaign <絶対path>`をCPUで起動すると、追跡と承認後poseを別々の
`all`ジョブとして共有training queueへ登録する。queue workerは別途稼働させておく。
レビューはGPU枠を持たず並行実行する。queue登録直後の中断も決定的なjob名で回収し、
再開時に二重投入しない。取消済みjobは勝手に再登録しない。
各GPU段階は最大試行数まで再試行し、3clip連続失敗でcoordinatorを停止する。
既に登録したjobの所有情報は保持する。同じorchestrate入口で再開する。

clipごとのartifactは次の順序で確定する。

- `generation.json` / `tracks.npz`: pose配列を持たないraw追跡と元検出row。検出・CLIP特徴はchunkごとに保存。
- `review.json` / `decision.json` / `selection.npz`: GPTの承認と選択された実観測。raw tracks/hashを固定して参照する。
- `pose.json` / `poses.npz`: 選択観測だけのViTPose。完了chunkとレビューhashを照合して再利用する。
- `publication.json`: poseとレビューの対応を検証した後の公開receipt。

レビュー指示の正本はWORKER.md。全frameのID overlay一覧と人物crop一覧を作る。
全raw観測の重複ない被覆、同一frameでのplayer ID衝突、duplicateの相手、入力hashと
画像一覧のcoverageを検証する。曖昧な区間はneeds_review、処理失敗はreview_failedで保留する。
レビュー承認だけではdatasetをapprovedにしない。非選手・重複・欠測・GSI補間には
poseを推論しない。元検出row・player ID・frame/PTS・欠測maskを全段階で保持する。
利用枠制限は30分待って再開する。

`status --campaign <絶対path>`は追跡待ち・レビュー待ち・保留・失敗・pose待ち・公開済みを区別する。
`metrics.json`にも全体／追加clipの状態別件数、新規の全検出数・pose crop数・段階別時間を保存する。
採用データは別datasetのmanifest.json、clips/*.npz、reviews/*.jsonに保存する。RGBは複製しない。
PlayerPoseStoreはball store hash・frame/PTS・artifact・レビュー・欠測maskを検証し、
未採用clipを黙って空poseにしない。skipだけNoneを返す。

単一clipの入口は`generate-clip`、`review-clip`、`pose-clip`、`publish-clip`（いずれも`--index`必須）。
GPUコマンドはtraining queue内からのみ実行できる。pose-clipはpose生成後に公開も行い、
公開直前に中断した場合はpublish-clipでGPU推論を繰り返さずに復旧できる。
テストは`tests/unit/tennis_scene/chat_annotation/player_pose`。実GPU/modelは使わない。
