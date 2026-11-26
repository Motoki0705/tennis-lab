# Meijiの全身骨格と排他割当のCPU評価

[#2444](https://github.com/Motoki0705/tennis-lab/issues/2444) のフレーム別3D骨格を、保存済み入力で評価した。検出・pose・ボールの再推論は不要で、CPUで復元から保存・読戻しまで実行した。 [実行入口と成果物schema](../../../src/tennis_scene/multiview_pose/README.md)

## Meiji 3camの人物対応と未使用視点への再投影

#964 r6/r7の全画面DINO `.30`・FP32 ViTPose全40,531行と、#933の校正・#932の人手確認ボールに基づくclip別half-turnを使った。試作と同じ全3,497元フレームを、同じ人手ラベルおよびIoU 0.5の既存採点器で照合した。この評価入口は、固定したカメラをソルバーに渡すためのものである。動画共有判定を読む通常入口はmeiji_4camで評価した。 [試作入力](../run-i2444-multiview-pose-probe/README.md#meiji-3camのフレーム別割当)

| clip | 元frame | 肩・腰の試作MILP | 全身の実装 |
|---|---:|---:|---:|
| `video_000/clip_000` | 1,010 | 0.7377 | 0.7415 |
| `video_000/clip_007` | 652 | 0.9292 | 0.8935 |
| `video_001/clip_001` | 1,300 | 0.8640 | 0.7760 |
| `video_002/clip_013` | 535 | 0.8933 | 0.8745 |
| 4 clipのmicro pair F1 | 3,497 | 0.8485 | 0.8050 |

TP/FP/FNは15,683/1,102/4,500から14,205/906/5,978になり、正解リンクの被覆低下がF1低下につながった。選手同士の組に限ったF1も0.8728から0.8258。選手同士の誤リンクは70から16に減った。精度と被覆の交換関係を示す開発clipの結果で、別収録への一般化は評価していない。 [clip別の採点・入力SHA・診断](meiji3.json)

各clipで3台を一台ずつ外し、残り2台だけで観測の候補生成・割当・三角測量をやり直した。全3視点で同じ選手のラベルboxと生検出が照合できるperson-frameを分母とし、外した視点に残る観測可能なViTPose関節を関節分母とした。人物対応が正しかった2視点候補から、外した視点の同一人物のViTPoseへ再投影した。外した視点のboxは、ラベルへのIoU照合と、重複時のDINO scoreで選んだ。

採点可能なのは15,458 / 19,860 person-frame（77.83%）と251,343 / 332,112関節（75.68%）。採点可能点での誤差は中央値9.59 px、P90 27.81 px、P95 59.39 pxだった。未使用視点のViTPoseとの整合を測るため、カメラ校正と両側の2D推定誤差を含む。 [clip・未使用カメラ別の分母と残差](meiji3.json)

## meiji_4camの被覆

#2363の採用済み検出・pose・校正と、#2241のボール由来動画共有判定で全69 clipを処理した。元動画実ファイルのSHA-256とカメラ行列を照合し、各clipの候補を別storeへ公開してschema検証付きで読み直した。

- 全46,228元frameで候補を生成した。採用候補348,542体の支持は2視点110,936、3視点25,776、4視点211,830。接戦maskは63,931体。
- 1,332,834検出行のうち1,146,520行を割り当てた（86.02%）。視点不足111,594、競合72,539、幾何不一致1,594、混入587行。1行に付く棄却理由は最後まで進んだ段階であり、混入の実正解件数とは独立する。
- #2363が停止した `clip_012`、`clip_020`、`clip_058` でも元frame全数で候補を保存した。ここでの被覆は候補が1体以上あるframeの割合を表す。
- 共有sideからの単独判定は51 clipが確定一致、18 clipがmargin不足の警告。全clipに固定した動画判定を適用した。

4 CPUプロセスでの再実行は329.00秒だった。clipごとの経過時間を合計すると、入力照合22.16秒、復元1,171.82秒、保存と読戻し22.32秒。clipごとの復元時間を元frame数で割った値の中央値は24.84 ms/frame。Meiji 3camの採点とLOVOは127.99秒。測定中に他のローカルCPU検証も実行した。

69 clipの被覆と入出力の版を保存した。 [全clipの計測](meiji4.json)

## 再現

リポジトリrootで実行する。#2363と共有コート・driftを置くworktree outputsのrootは `MEIJI_POSE_SHARED_OUTPUTS`、#932・#933・#964のoutputsを置くE側rootは `MEIJI_POSE_ARCHIVED_OUTPUTS` で上書きできる。meiji_4camは4 CPUプロセスで実行し、元fpsの全フレームを処理する。

```bash
bash knowledge/runs/run-multiview-pose-meiji-20261124/reproduce.sh meiji3
bash knowledge/runs/run-multiview-pose-meiji-20261124/reproduce.sh meiji4
```

実行結果の既定は `outputs/i2444/acceptance/{meiji3,meiji4}/metrics.json`。4camのclip storeは `outputs/i2444/acceptance/meiji4/video_000/<clip_id>/`。JSONの `artifact_id` と `descriptor_sha256` で保存版を照合できる。初回と再実行で4cam全69 clipの配列SHA、3camの精度集計が一致した。`clip_012` は通常CLIからの公開でも再実行と同じartifact ID・記述子SHAになった。撮影データと骨格配列はローカルに保持する。
