# Meijiのフレーム別割当で正解リンクを回復

[#2444](https://github.com/Motoki0705/tennis-lab/issues/2444) の全身候補で、肩・腰の試作から減った正解リンクを追跡した。比較は同じMeiji 3cam開発用4 clip・元3,497フレーム・同じ全画面DINOとViTPose、固定カメラ、IoU 0.5の採点器を使った。 [初回評価の条件と入力の版](../run-multiview-pose-meiji-20261124/README.md)

## 取りこぼしの内訳

試作だけが拾った正解リンクは2,458件、初回の全身版だけが拾った正解リンクは980件。差し引きでFNが1,478件増えた。

| clip | 骨長 | 全身残差 | 両方 | MILP競合 | 失った計 | 新たに拾った |
|---|---:|---:|---:|---:|---:|---:|
| video_000/clip_000 | 475 | 6 | 0 | 99 | 580 | 573 |
| video_000/clip_007 | 134 | 4 | 0 | 175 | 313 | 92 |
| video_001/clip_001 | 525 | 21 | 1 | 775 | 1,322 | 178 |
| video_002/clip_013 | 146 | 2 | 2 | 93 | 243 | 137 |
| 合計 | 1,280 | 33 | 3 | 1,142 | 2,458 | 980 |

同じ正解リンクを含む生の検出組をすべて調べ、候補生成・支持視点・全身検査を通った組が一つでもあればMILP競合と数えた。検査で全組が落ちたリンクでは、支持を満たす最小視点数の候補で残差と骨長を判定した。候補生成と支持視点不足が原因のリンクは各0件だった。骨長違反は肩幅1,016、腰幅922件のフラグが多かった。両方に該当するリンクを含む。

近側・遠側は、カメラ・フレームごとにラベル上の両選手のbox下端の中点で分けた。失ったリンクは遠・遠1,443、遠・近1,004、近・近10、遠・不明1件。既存採点器で正解リンクになるのは同じ選手の組で、非選手を含むリンクはFPに数える。FPを選手同士／非選手を含む組に分けると、試作は70／1,032件、初回全身版は16／890件。clipと役割、近・遠の交差集計を保存した。 [内訳と入力SHA](losses.json)

## 設定の選択

頭部・手首・足首に局所マスクを許すだけではF1 0.8127、FP 944件だった。骨長を視線間角度で補正し、全身の支持と残差で割り当てる設定はF1 0.8192、FP 1,187件だった。初回の支持条件を残して肩・腰のスコアの重みを増やしても、試作のF1とFP上限を同時には満たせなかった。

肩・腰のペア三角測量に試作と同じ24 px相当の残差上限、1度の視線間角度、コート面から−0.2〜3.0 mの範囲を使い、そのスコアでMILPの競合を解いた。全身の関節は高い確信度の観測間の残差と骨長で検査する。胴体で骨長を超えた場合は、各違反骨の端点を少なくとも一つ無効にし、無効化する関節数が少なく支持の多い組を残す。採用には骨格全体で有効な肩・腰が2関節以上、かつ各カメラで有効な関節の支持が必要。

腕・脚の付け根側の骨は、元画像1080p換算の8 pxを距離・焦点距離・視線間角度から位置の誤差幅に換算し、その余裕を含めても長すぎる場合に候補ごと棄却する。全身スコアでの補正量4・8・12 pxの開発用比較で8と12の値が一致したため、8 pxを固定した。同じマスクと対応スコアで骨格全体の棄却を省いた条件はF1 0.8510、FP 1,050件だった。位置誤差を見込んだ付け根側の骨長検査を入れるとF1 0.8515、FP 1,045件となり、複数視点で別人の腕に一致する合成反例も棄却できた。比較と選定は開発用4 clipに限定した。 [開発比較の実測](dev-ablations.json)

## 固定後のMeiji 3cam

| clip | 肩・腰の試作 | 初回全身版 | 改善後 | 改善後のFP／FN |
|---|---:|---:|---:|---:|
| video_000/clip_000 | 0.7377 | 0.7415 | 0.7638 | 24／2,234 |
| video_000/clip_007 | 0.9292 | 0.8935 | 0.9313 | 31／445 |
| video_001/clip_001 | 0.8640 | 0.7760 | 0.8538 | 968／1,180 |
| video_002/clip_013 | 0.8933 | 0.8745 | 0.8948 | 22／587 |
| micro pair F1 | 0.8485 | 0.8050 | **0.8515** | **1,045／4,446** |

試作に対してTPは15,683→15,737件、FPは1,102→1,045件。改善後のFPは選手同士7件、非選手を含む1,038件。選手同士の組だけで集計したF1は0.8761だった。

各視点を順に外した再実行で、16,662／19,860 person-frame（83.90%）、267,019／332,112関節（80.40%）を採点した。再投影誤差は中央値10.02 px、P90 43.64 px、P95 101.44 px。初回は採点可能関節75.68%、中央値9.59 px、P90 27.81 px、P95 59.39 pxだった。採点できる関節が増え、残差の裾は悪化した。各設定の採点可能点全体で集計したため、共通点だけでの変化は未計測。 [clipごとの精度と未使用視点](meiji3.json)

## meiji_4cam全69 clip

入力の元動画、検出行、pose、校正、#2241のボール由来の向きを初回と同じ版で照合した。全69 clip・46,228元フレームで候補を保存して読み戻した。

| 指標 | 初回全身版 | 改善後 |
|---|---:|---:|
| 候補ありフレーム | 46,228／46,228 | 46,228／46,228 |
| 生成候補 | 348,542 | 342,692 |
| 割当済みDINO行 | 1,146,520／1,332,834 | 1,144,875／1,332,834 |
| 行の割当率 | 86.02% | 85.90% |
| 4 CPUプロセスの経過時間 | 329.00秒 | 367.68秒 |
| clipごとの復元時間の中央値 | 24.84 ms/frame | 28.55 ms/frame |

開発用評価・pytestと並行した試行の経過は571.58秒だった。このworkerの他の評価を終えてから同じ4プロセスで再計測し、表に採用した。2回の成果物のID、記述子SHA、候補数および行の診断は69 clipすべてで一致した。旧測定は当時の同時負荷を含む。 [clip別の版と再計測結果](meiji4.json)

#2442の疑似正解は品質を理由に棄却されたため、対応の採点を取り下げ、ダブルスの精度は新旧の横並び比較でユーザーが定性評価する。 [比較ページ #2459](https://github.com/Motoki0705/tennis-lab/issues/2459)

比較に使う候補は、このworktreeの `outputs/i2444/refine/timing/meiji4/<video>/<clip>/scene.json` から読み込む。

## 予約した未見3 clip

#964 r18には3 clipとも全画面DINOと人手の部分参照ラベルがあり、追跡前のViTPoseのポートは全9カメラで欠けていた。`video_000/clip_002` と `video_002/clip_003` は固定されたボール参照と3視点の校正があり、`video_001/clip_003` はcam0校正とボール参照の向きが欠測していた。未見の採点には実装 `466ba35cfacc496a9b8b895de2f85fad6a8dd849` と `meiji3.json` の `config` を固定した。 [予約入力の点検](holdout-inputs.json)

`video_001/clip_003` には、9フレームから作った別runのCourt投影点が残っていた。3本の動画SHA-256を照合し、cam0の14投影点から既存の平面pinhole fitで校正を復元した。RMSEは3.13 px、既存の校正上限は11.01 px。cam1・cam2はr18の校正を使った。このRMSEは保存済みhomographyからの投影点に対する値で、Court診断で元の検出点と比較したP95残差は15.84 pxだった。

同じclipのボール注釈から実際に見える点を選び、#932の設定で1,249フレームを採点した。cam0を基準にcam0／cam1／cam2のhalf-turnは `[false, false, true]`、最良仮説の支持は91.35%、次点との差は0.475。親は2026-11-24にこの校正と向きを予約評価に使うことを承認した。 [保存済みCourt投影点](../run-slcs-meiji-court-visual-qc-v1/results.json) [復元した行列とボール判定](holdout-camera-recovery.json)

共有GPUロックのACL復旧後、ジョブ `0-8204503069350331066_2026-11-24T05-08-50-649668933Z_150553_i2444-holdout-vitpose.job` が完了した。開発用と同じcheckpoint・FP32・flip test・bbox拡張率で、3 clip・9カメラの38,421検出行にViTPoseを付与した。固定計画および9視点の元動画・検出行・成果物のジョブ由来を照合した。 [固定計画と準備時の検証](holdout-pose-preparation.json)

#2445に渡す入力はこのworktreeの `outputs/i2444/holdout/vitpose/stores/<video>/<clip>/scene.json` で、`person_detections` と全行の `person_pose_features` を持つ。完了記録は `outputs/i2444/holdout/vitpose/complete.json`。SHA-256は `4966ade3bec4fea665a2613f1e9255117de70f1f2a58fa12aa7ccef0ecbe7270`。

### 人手の部分参照ラベルでの採点

取得したposeを固定設定で一度だけ採点した。IoU 0.5以上で照合したr18の部分参照に基づいて、フレームごとのカメラ間対応を数えた。全ラベルの指標では非選手を含むリンクもFPに数える。選手同士の指標も同じ採点から集計した。

| 対象 | 照合したラベルbox | micro pair F1 | 選手同士のmicro pair F1 | TP／FP／FN（全ラベル） |
|---|---:|---:|---:|---:|
| video_000/clip_002 | 14,088／14,088 | 0.7724 | 0.9011 | 6,465／2,394／1,416 |
| video_001/clip_003（校正補完） | 15,399／15,399 | 0.7671 | 0.8512 | 7,016／1,824／2,436 |
| video_002/clip_003 | 8,645／8,645 | 0.6808 | 0.8110 | 3,028／1,433／1,406 |
| **全3 clip** | **38,132／38,132** | **0.7517** | **0.8621** | **16,509／5,651／5,258** |
| **補完clipを除く2 clip** | **22,733／22,733** | **0.7406** | **0.8703** | **9,493／3,827／2,822** |

採点時のHEADは `590c2377810762fdea0d5485d10307e612de51eb`。CPUの経過時間は61.49秒、終了値0。採点開始記録と結果の入力SHAを照合し、2集計のTP・FP・FNがclip別の和と一致した。 [採点結果と入力SHA](holdout-pair-score.json)

### 実行コマンド

ViTPoseは、同じworktreeで4ファイルの読み書き権限と親が管理する共有queueの稼働を確認して投入した。 `CODEX_THREAD_ID` には投入時のセッションIDを使った。

```bash
(
  for lock in /var/lib/tennis-lab-actions/gpu.lock{,.gate,.slot-0,.slot-1}; do
    if ! { [ -r "$lock" ] && [ -w "$lock" ]; }; then
      printf 'GPU lock is not readable and writable: %s\n' "$lock" >&2
      exit 1
    fi
  done

  TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue \
  bash .agents/skills/training-queue/scripts/training_queue.sh add \
    'PYTHONPATH=.:knowledge/runs/run-i2444-multiview-pose-probe .venv/bin/python -B knowledge/runs/run-multiview-pose-link-recall-20261124/holdout_pose.py infer --plan outputs/i2444/holdout/pose-plan.json --plan-sha256 6043c92a2f0cb58a5e3583bb6230e75d40e5695de9e40549f47e5d4ca8d33be9 --output outputs/i2444/holdout/vitpose' \
    --name i2444-holdout-vitpose --provider codex --session "$CODEX_THREAD_ID" \
    --issue 2444 --resource all --idempotency-key i2444/holdout-pose/acl-recovered
)
```

親CLIが完了を確認した後、以下のコマンドで採点した。採点器は `vitpose/scoring-attempt.json` を排他的に作成し、`pair-score.json` に両集計を保存した。

```bash
bash knowledge/runs/run-multiview-pose-link-recall-20261124/reproduce.sh holdout-score outputs/i2444/holdout
```

## 再現

リポジトリrootから実行する。既定出力は `outputs/i2444/refine/qualified`。保存済みデータの配置と環境変数は初回評価と同じ。

```bash
bash knowledge/runs/run-multiview-pose-link-recall-20261124/reproduce.sh trace
bash knowledge/runs/run-multiview-pose-link-recall-20261124/reproduce.sh meiji3
bash knowledge/runs/run-multiview-pose-link-recall-20261124/reproduce.sh meiji4
bash knowledge/runs/run-multiview-pose-link-recall-20261124/reproduce.sh holdout-inputs
bash knowledge/runs/run-multiview-pose-link-recall-20261124/reproduce.sh holdout-recovery outputs/i2444/holdout
bash knowledge/runs/run-multiview-pose-link-recall-20261124/reproduce.sh holdout-plan outputs/i2444/holdout
```

公開するのは再現コードと指標・digest。元映像と骨格配列はローカルに保存する。
