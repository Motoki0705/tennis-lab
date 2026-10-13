---
id: group-i983-court-throughput-s42
type: group
task: court_detection
sequence: 42
recorded_at: '2026-10-12'
title: Court学習の時間内訳とbatch8/16の比較 (#983)
issue: 983
members:
- run-i983-court-throughput-b8-s42
- run-i983-court-throughput-b16-s42
parents: []
papers: []
tags:
- issue-983
- colab-l4
- throughput
config:
  scope: Short S+ timing diagnosis, separate from the 20-epoch accuracy comparison
metrics:
  batch16_images_per_second_gain_percent: 3.0314815740769774
artifacts:
  run_dir: knowledge/runs/group-i983-court-throughput-s42
  training_wall_time: knowledge/runs/group-i983-court-throughput-s42/training-wall-time.json
  environment: knowledge/runs/group-i983-court-throughput-s42/environment.json
  input_snapshot: knowledge/runs/group-i983-court-throughput-s42/inputs.json
  input_verification: knowledge/runs/group-i983-court-throughput-s42/input-verification.json
relations:
- to: run-i983-court-dinov3-splus-s42
  rel: analyzes
---

ユーザーの「Driveからデータをローカルへ取得していないか」「モデルが大きくbatchを増やせず遅いか」という仮説を調べた。元S+ runの準備記録と、同じGit/入力を復元した新しいL4の実体を照合した。入力は `/content/tennis-lab/data` の通常のVM filesystemへ取得済みで、学習batchごとにDriveから画像を読んでいない。

| 短時間診断 | 秒/step | 画像/秒 | peak allocated GB | peak reserved GB |
|---|---:|---:|---:|---:|
| batch 8 | 0.7953 | 10.0588 | 9.095 | 9.890 |
| batch 16 | 1.5438 | 10.3637 | 16.157 | 17.450 |

batch16はOOMせず動作したが、画像/秒の改善は3.03%だった。batch8は検証済みの上限値ではなく、4条件比較で事前に固定した条件である。大きなbatchほどstep/秒が増えるわけではないため、固定画像数の処理量で比較する。この結果をViT-Lのbatch16容量や長期学習の収束保証へ一般化しない。

batch8の32stepではデータ待ち約0.42ms、H2D約15.95msに対し、forward/loss約168.39ms、backward約517.16ms、clip/optimizer約57.59msを要した。train metricsは約35.82ms（全stepの約4.5%）だった。約500のGPU結果CPU読取箇所というコード上の候補はあるが、今回の遅さの主因と断定する根拠にはならない。GPU上のbatchを使い回しても10.06→10.30画像/秒に留まり、入力供給が主な制約という仮説を支持しなかった。

DINOv3は完全凍結されている一方、比較のためTransformer/DPT/headsを全variantでViT-L幅に固定している。共通の学習parameterは128,635,436で、S+にも独立射影が加わる。backwardの中心はこの学習する後段であり、小さいbackboneだけを選んでも学習計算全体が小型化する構成ではない。GPU使用率約92%という観測と時間内訳は、今回の固定構成では計算側が主な制約であることを支持する。

S+の本学習は33,160 stepで、1step/秒でも約9.21時間になる。完了receiptとstdoutの時間内訳は次の通りである。

| 区間 | 時間 | 解釈 |
|---|---:|---|
| 学習progressの合計 | 9:06:20 | data/model/loss/metrics/loggingと学習中の同期待ちを含む。純GPU時間ではない |
| validation 6回 | 0:20:43 | 各394 batch |
| validation後の追加tail | 0:15:47 | checkpoint/callback等。個別内訳は未計測 |
| 終端testと後処理 | 0:25:42 | 111 batch推論1:40、連結・圧縮・同期等24:02 |
| 上記へ配賦できない時間 | 約0:52:24 | epoch境界、起動、最終同期等を分離できない |
| job全体 | 11:00:56 | 開始からDrive最終保存まで |

checkpoint保存と60秒ごとのArtifactStore同期は学習プロセス内で待つ実装である。S+の約1.655GBのcheckpointをlast20回・validation世代6回保存するため、公開データ量は概算約43GBになるが、実通信量や各転送時間は測定していない。終端testは全予測の連結とNPZ圧縮を行い、推論の100%表示から保存までの時間も長い。正式なbest checkpoint評価はこの終端testとは別jobである。

改善候補は、必要なcheckpointを保持した安全な保存待ちの短縮、正式評価と重複する終端test全配列出力の必要性の見直し、同じ指標を保つmetrics集計のベクトル化である。batch増大だけで大幅な短縮ができる観測ではない。学習量・モデル容量・評価条件を変える施策は別実験として相談する。今回の4条件はbatch8・20epochのまま維持し、速度診断を精度結果に混ぜない。
