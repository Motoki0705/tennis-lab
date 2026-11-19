#2227 を取り込んだ `main`（`615e132ca`）へ載せ替え、#1755 の差分33ファイルに絞りました。ブランチは `codex/video-batch-batching`、HEADは `decb7dc13e6de3331fb902498ddcd9fc6792d579` です。PRはReadyのままです。[PR #2272](https://github.com/Motoki0705/tennis-lab/pull/2272)

- CPUの小型X3D（24×40、T32）で、B=2と末尾B=1のforwardが各1件だけ記録されることを確認しました。
- バッチ有効時に、資産の差し替えをモデル読込前に拒否し、共通の最終化で共有court・drift・sideの破損を拒否しました。同じ世代への同時exportは直列化され、両clipを公開できました。
- 最終HEADで関連CPUテスト823件、pre-commit、Python CI全14ジョブ、Knowledge CI両ジョブが成功しました。ローカルの検証プロセスは終了済みです。[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44500809634)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44500809628)

今回の検証はCPUのみです。実GPUでの横断バッチの出力一致、速度、メモリの測定は #1757 の条件3に残っています。[計測PR #2273](https://github.com/Motoki0705/tennis-lab/pull/2273)