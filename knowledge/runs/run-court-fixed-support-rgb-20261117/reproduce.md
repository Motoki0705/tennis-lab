# 固定支持点集合とRGB塗線中心の受入比較

実装は `49f605aab`。`protocol.json` は全件評価前に固定した設定・source/reference SHA256であり、評価scriptは開始時に照合する。
入力はrun70の凍結済みraw、参照はrun71の承認済み白線H。参照/H/許容差は変更しない。
GPU再開と実装の承認は #2150 の親note `6920971828`。

## 実行

worktree rootで実行する。Pythonのimportはこのcheckoutを使う。
数値評価は`49f605aab`をロードしたprocessで実行した。その後の`56a0e8812`は診断の保存形式だけを変更している。
`source-equivalence.json`に数値module・参照・設定・採否/順位付けの不変性を記録し、公開APIの8条件の一致も再確認した。
厳密な再現では、最終PRのrun bundleと必要なoutputs資産を持つ**変更のない再現用worktree**で、
下の1ファイルを評価時の版へ戻してから実行する。現在の作業中worktreeや人間の変更がある場所では行わない。

```bash
git restore --source=49f605aabbea65dc09c00c28a72a02b4a4ea59b2 -- src/tasks/court_detection/inference/regions.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/materialize_rgb.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/evaluate.py --dataset tennis_clip --limit 1 --workers 1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/evaluate.py --workers 10
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/summarize.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/diagnose_results.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/render.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/figures.py
git restore --source=HEAD -- src/tasks/court_detection/inference/regions.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/api_parity.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/api_parity.py --accepted-only
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-fixed-support-rgb-20261117/check_evidence.py
```

`outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117/` に大きな配列・全case診断・原解像度画像を保持する。
rawのKP heatmapとLINE logitsは元cacheにない。品質境界が消費するdecoded KP/score/valid/LINE probabilityだけを渡し、未保存fieldは空tensorとして明示する。推論モデルの新しい出力を作ったとは扱わない。

同一画像・領域でLINE evidenceを一度作り、旧処理と新処理およびモデル品質判定で同じ配列を共有する。
`line-evidence/`にはdistanceだけでなく全ridge点・抽出点・重み・接線・coherenceを保存し、再開時には再抽出しない。
caseの結果が存在すれば読み出す。別の設定での追試には別run/outputが必要。
raw/LINE/原画像を欠いた場合に他の入力へ切り替えない。

`legacy`はrun70にversion管理された旧solverと旧採否（全点が画像内・凸・面積・inlier>=8）。
`fixed_support`は(a)とモデル内部quality、`fixed_support_rgb`は同じ(a)に(b)とRGB採否を加える。
RGBはcropのモデルHを元画像座標へ移してから、元RGB1frame全体でfitする。
すべての候補・採否が確定した後に参照/GTを読み、誤差を求める。
不合格のRGB fitted Hは正しさを分類する診断候補であり、下流に渡す採用Hではない。

meiji_4camは元1920×1080で平均10px・最大25px。固定目標はFP0、正しい診断候補の採用率90%以上。
224画像の内訳は、参照のclip000/017/034/051/068とpilotの005/013/022/030/039/047/056/064を除いた56clip×4camera。
JSONの`holdout`はこの部分集合の識別名である。全276件のraw/candidateは過去の原因分析で参照しており、完全に未知の評価集合を意味しない。
他domainの正しさクラスは従来同様の対角比換算を補助値として出す。native px誤差・成立数・採用数・同一GT点でのpaired誤差が回帰の主指標。

`protocol.region_config`は`assess_region_prediction`へ渡す品質引数で、`enabled`はこの関数では参照されない。
評価の領域集合は凍結manifestで明示し、実pipelineの`enabled=true`と同じ`region_proposals`であることを`api_parity.py`で照合した。
4camera×model/RGBの8条件で、公開`select_court_region`とオフライン採否・選択領域の一致を確認する。model forwardだけは凍結headを使う。
`api_parity.py --accepted-only`はclip068/cam0のRGB採用経路を追加検証する。`check_evidence.py`は旧/新solver前後の共有LINE配列の内容hashを比較する。
`archive.py`は小さな表と検証記録をrun bundleへ写し、大きな入力・LINE・全case・画像にはoutputs側の内容inventoryを作る。

## GPU起動の失敗

`capture.py`は共有queueへhalf/provider codex/issue2190、session `01a1e30b-a305-7672-8355-525787fd6380`で登録した。
`capture-failed.job`と`capture-failed.log`が正確なコマンドと失敗記録。
queueは `/var/lib/tennis-lab-actions` のアクセス権により `lock-namespace` で停止し、モデルのload/forwardは起動していない。
training-queue skillの規定に従い共有lockを修復・迂回しない。新raw取得やCUDA実機確認は未実施。
保存したjobは履歴であり直接実行しない。復旧後の再開には共有queueへ新しいattempt keyで登録する。
このためCPU再現では元の凍結rawを使い、全3,545画像の元RGBをSHA256で検証した。

## 検証

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/unit/tasks/court_detection/geometry tests/unit/tasks/court_detection/inference tests/unit/tennis_scene/pipeline/components/test_court_kp.py tests/unit/tennis_scene/test_pipeline_configuration.py -n 4 --durations=0 -q
.venv/bin/pre-commit run
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py --check-summary --base-ref origin/main
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_repro_paths.py
```

人工画像は既知Hから独立にforward描画し、paint・net・遮蔽物・画面外を含む。RGB不確かさgateを無効化する一時変異は専用nodeのassertで検出し、元のsource bytesへ戻して同node成功を確認した。
最終Hと選択集合/残差/終了理由、RGB失敗時の無効mask、crop offset、schema、設定のcache identityも検証する。
CUDA bf16が必要な既存1nodeはCPUではskipであり、成功へ数えない。
