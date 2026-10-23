# 新規視点の小規模品質診断の再現

結果・解釈は[runノード](../../nodes/synthetic_data_generation/000025-run-court-novel-view-quality-20261022.md)、
採用前の判断依頼は[設計文書](../../reports/court-scale-trajectory-design.md)。
常設の生成器ではない。既存src/presetを変えず、前回の
[CPU prototype](../run-court-nested-plan-cpu-20261022/prototype.py)を明示的に再利用する。
`cases.json`の見出しは人手で選んだ定性例で、品質の正解ラベルではない。

## 入力と条件

- tennis-lab `53e5a6336` のsrc、NHT `9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6`。
  NHTのpublic CLIだけを呼ぶ。`spin setup-nht`を実行しない。
- `inputs.json`は各sceneのalignment、scene/cameras、保存config、既存dataset descriptor/index、
  checkpoint、renderer configの計32ファイルのSHA-256。
  B01旧configの欠損keyをCPU prototypeと同じくメモリ上で明示補完する。元の資産は変更しない。
- 推奨と3つの代替の数値は`policy.json`。
  D2-aは拡張率だけを5%へ変え、旧保存preset全体を再現するものではない。
  D2-bはHFOV範囲を3層45/67.5/90度で代表する。旧選択器の再実装ではない。
- 各scene/条件でcomplex中心のcircleを96点、6軌道×3HFOV層の各層からtarget KP数の
  min/maxを選んだ36点。いずれも幾何のみで描画前に選び、悪いRGBを見て差し替えない。
  2点が連続列と重なるので実描画は130点。意図的な標本で母集団の採択率を推定しない。
- 1m jitterは面積一様disk。点IDはscene・軌道ordinal・既約弧長割合・HFOV層から作る。
  条件、保存順、動画frame番号をseedへ混ぜない。各条件の同じ点は同じ乱数を使う。
  D2-aは各曲線の弧長割合を対応付ける。拡張で長さが変わるので位置は一致しない。
  D2-bは位置・pose一致、D6は位置・K一致を`pair-checks.json`で検査する。
- productionではpre-render gateで落ちる候補も診断のため実描画する。
  near/far曖昧点のv3投影はnull相当とし、KPを推測で補完しない。

## QAの定義と保存先

`analyze_probe.py`は各RGB/alpha/depthを`NHTRenderArrays`で検証し、
既存`project_court_semantics_v3`と`attach_renderer_visibility`を使用する。
alpha/depthの単位はpublic NHT座標。正depth判定はmetric変換で変わらず、
深さの絶対値・期待court depthとの差をgateにしない。

- 穴: alpha<.01、depth≤0、またはその和集合。それぞれ全体とtarget court ROIで集計。
  α<.5/.9の面積率、正depth率、alphaのp05も残す。
- ROI: physical corner 0/1/3/2をcamera near=.01mで3D clippingした後、画面でclipしたpolygon。
  画面外のROIを外挿しない。投影不能・空ROIはnull。KPが0でもコート面の一部が写ればROIはあり得る。
- 既存gate: 全court合計のgeometry KP≥4かつrenderer KP>0、near/far曖昧は棄却。
  renderer KPは各KPの3×3近傍にα≥.01かつdepth>0があるかという既存契約。
- 追加候補: target renderer KP≥4、ROI穴≤1/5/10%。別の感度診断として
  ROIのα<.9面積≤1/5/10/25%を比較。採用値ではない。
- coverage: 既存実装のgeometry KP数でfull=14、near_full=10〜13、partial=4〜9。
  α由来のrenderer支持とは別。geometry/renderer KP数は両方残す。
- 白線の簡易値: RGB grayscaleの15px openingとの差>.06、max RGB>.45、RGB幅<.30の
  明るいridgeを抽出。投影した地上court skeletonのvalid/ROI内pixelから最近ridgeまでの距離を
  24pxで打ち切り、中央値と3px以内率を記録。暗い/細い線・照明・ぼけ・背景を分離できず、
  true white-line correspondenceも独立alignment GTもない。条件間で線幅も変わるためhard gateにしない。

全frame値、完全な層別JSON、raw配列は`tmp/court-render-quality/run-01/`に残す。
Gitにはscene/条件の分位点`summary.json`、層別の小さな`strata.csv`、
gate後のcoverage/HFOV/shape/target分布`gate-distributions.json`、小さい代表図だけを保存する。
動画・contact sheetの完全なパス、サイズ、hashは`media-manifest.json`。
現行ローカル閲覧入口は次。

```text
tmp/court-render-quality/run-01/review/index.html
```

動画は各96 frame・12fps・8秒。RGB左/overlay右、比較動画は推奨上/代替下。
HFOV比較動画は92.5対67.5度だけで、端の75/110対45/90度はcontact sheetとQAで確認する。
フレーム間隔は弧長割合であり、カメラの物理速度や撮影時刻を定義しない。

## 再実行

このcheckoutをcwdにし、インストール済み`nht-render`を使う。入力は読み取り専用。
出力はこのworktreeの新規`tmp/`内ディレクトリへ置く。以下の環境変数を設定する。

```bash
export SCENES_ROOT=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes
export PROBE_OUTPUT_DIR="$PWD/tmp/court-render-quality/reproduce-01"
export PROBE_REPLAY_DIR="$PWD/tmp/court-render-quality/reproduce-plan-01"
export TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue
export PROBE_SESSION="$CODEX_THREAD_ID"
bash knowledge/runs/run-court-novel-view-quality-20261022/repro.sh plan
bash knowledge/runs/run-court-novel-view-quality-20261022/repro.sh enqueue
```

`enqueue`はGPU処理を直接起動しない。queueのworkerが停止中なら、同じcwdから起動する。
他のworkerが動いている場合は既存workerが処理する。CPU QAはGPU描画完了後に実行する。

```bash
bash .agents/skills/training-queue/scripts/training_queue.sh start
# queueの完了を確認した後:
bash knowledge/runs/run-court-novel-view-quality-20261022/repro.sh analyze
```

容量probeはbatch1の1枚→batch4の4枚。各scene/条件をbatch4で描く。
メモリ監視は0.5秒ごとのdevice全体使用量で、renderer単独の厳密なpeakではない。
10000MiB超を観測したら、その呼出し終了時に後続へ進まず停止する。
これはVRAMの強制上限ではないのでbatchを勝手に増やさない。queueは`--resource all`を使う。
renderer失敗・既存出力との衝突・不一致は例外で停止し、再生成で既存成功結果を上書きしない。
描画のみの完了時間とメモリは`render-timings.json`、CPU解析の時間は`analysis-time.txt`へ残る。

再実行時のsmall evidenceは新しい出力内`evidence/`へ保存し、歴史的なrun bundleを上書きしない。
原runのpromoteだけは`review_bundle.py --evidence-output <このbundle>`を明示した。
原runの正確なNHT呼出しargvは`render-timings.json`と`smoke-*-timing.json`、
src/NHT版・条件は本READMEとrunノード、通常検証は`validation.txt`を参照する。

## 中断後の残存成果の確認

`verify_evidence.py`は入力32ファイル、既存の再計画36ファイル、mediaのhashと動画全frame、
保存QAの候補/描画/採択計数を照合する。rendererは起動せず、元の出力を上書きしない。
この監査はRGB/alpha/depth配列の再QAではなく、残存ファイルと保存済みQAの整合検査である。
bytecodeはknowledge bundleへ残さない。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  knowledge/runs/run-court-novel-view-quality-20261022/verify_evidence.py \
  --scenes-root /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes \
  --plan-root tmp/court-render-quality/run-01 \
  --replay-root tmp/court-render-quality/plan-replay-01 \
  --output tmp/court-render-quality/run-01/resume-audit.json
```

原成果の監査結果だけを`resume-audit.json`へ保存した。D2/D5/D6の最終採用は#1352のUI比較で決める。
