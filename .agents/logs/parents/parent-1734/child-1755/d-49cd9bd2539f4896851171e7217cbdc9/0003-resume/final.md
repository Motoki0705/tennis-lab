正式レビューのmajor 4件を修正しました。ブランチは `codex/video-batch-batching`、HEADは `4c79965c84a4daea9f3c3c53d85c72aff71c1b36` です。PRはReadyです。[PR #2272](https://github.com/Motoki0705/tennis-lab/pull/2272)

- ViTPoseは無効関節をUDP補正から除外しました。実デコーダーで、B=1と別clipが混ざるB=2の座標・信頼度が完全一致しました。
- Courtはlogitsとraw poseの非有限値を行単位で分離します。同じforwardにNaNを注入し、正常なclipの成果物が公開されることを確認しました。
- X3DはFP32のRGB+MDD前処理をforward前に予約します。720×1280・T32のB=1を4 sourceで処理する見積もりが約2.24 GiBになるため、既定上限を2 GiBから3 GiBに変更しました。500 MiBと1,500 MiBではpredictorへ到達する前に停止しました。
- ClipStoreはキャッシュヒット時にも配列のchecksumを検証します。同じサイズのまま値を破損させると拒否し、修復後に読み直せることを確認しました。

最終HEADで関連CPUテスト975件、pre-commit、Python CI全14ジョブ、Knowledge CI両ジョブが成功しました。ローカルの検証プロセスは終了済みです。[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44509575348)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44509575276)

今回の検証はCPUのみです。実GPUでの横断バッチ比較は #1757 の条件3に残っています。[計測PR #2273](https://github.com/Motoki0705/tennis-lab/pull/2273)