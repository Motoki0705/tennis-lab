#2363 のドラフトPRを作成しました。branch は `codex/parent-2278-issue-2363-primary`、HEAD は `1f4929edf` です。 [PR #2433](https://github.com/Motoki0705/tennis-lab/pull/2433)

**全69クリップをTRT FP16で試行し、55件が `generated`、14件が `failed`、0件が `blocked` でした。** 人物対応で13件、コート向きの判定で1件が停止しました。全件生成の受入条件は未達です。停止の根拠を記録し、14件の扱いと採否を親の統合レビューに返します。 [全件の診断表](knowledge/runs/run-i2363-trt-player-video/clips.csv)

- **速度・資源**　再構築は12時間10分58秒、Peak VRAMは1.76 GiB、Peak RSSは6.16 GiBでした。GPU遊休中の推定CPU時間は、プレーヤーの三角測量4,372秒、人物追跡776秒、入力組み立て670秒の順でした。 [モデル・stage別の内訳と改善候補](knowledge/runs/run-i2363-trt-player-video/README.md#時間と資源)
- **GVHMR・バッチ**　有限加算maskへの書換え前後で、実行に使った64 shapeのORT出力がビット一致しました。L=177を含む実GPUの6 shapeで出力の有限性とFP16層精度を確認しました。評価設定はViTPose-H 8 crop、HMR2 16、DINO 1、CLIP-ReID 16、GVHMR 4です。 [バッチ選定の根拠表](knowledge/runs/run-i2363-trt-player-video/README.md#バッチサイズの根拠)
- **記録の検証**　代表clipでは計時用CUDA同期で739.79秒から899.28秒に増え、成果物のバイト列・artifact ID・execution keyは一致しました。関連40テスト、変更した製品コードのpre-commit、knowledge検証が通りました。 [代表clipの計測](knowledge/runs/run-i2363-trt-player-video/qualification-v2.json)
- **品質・可視化**　生成分の3D関節点は96.04%、SMPLは95.79%が有効でした。低有効率の6件では、人物IDが1視点にしか紐づかない区間が主な欠測要因でした。代表1件、低有効率6件、停止14件の動画を生成し、全21本のデコードを確認しました。 [knowledgeレポート](knowledge/nodes/tennis_scene/000083-run-i2363-trt-player-video.md)・[動画一覧](knowledge/runs/run-i2363-trt-player-video/videos.json)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2433,"head":"1f4929edf86775e87bdad3a31270455cb6c6f59a","resume_on_success":true,"next_step":"ドラフト解除と最終報告"}}