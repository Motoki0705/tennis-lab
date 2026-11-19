# 白線中心の仮参照と再採点

Issue #2190 / PR #2200。production sourceは`b533eb7abdcb4fe7bdd125d41e2675cd75ceef4c`のまま。
このbundleに保存したoffline scriptを用いた。GPU不要。参照は未再承認であり、正式な受入合格の記録ではない。
親の明示指示により仮参照での採点だけ実施した。平均10px・最大25px、FP0・正しい採用90%以上は変更しない。

## 入力

- `data/tennis_multiview/processed/meiji_4cam/dataset/dataset.json`と各clipの`media/camN.mp4`。
  clip000/017/034/051/068、frame0/中央/末尾、各camera15frame。元1920×1080。
- 初期参照は以前のbundle `knowledge/runs/run-court-acceptance-implementation-20261116/references.json`。
  当時の人間承認を履歴として保持し、新参照の承認には引き継がない。
- 採点入力は以前のoutputsの`court-acceptance-implementation-20261116/meiji4-evaluation/meiji_4cam/**/evaluation.json`全276件。
  後から作られた`scene-evaluation`はclip033/cam0のB/Cでregion選択が違うため使わない。
  正本の旧mean/正誤を全件assertし、採否/region/points/qualityを一切変えない。
- 元データや凍結予測が欠けていれば停止する。別datasetや推論し直したrawへの切替はない。

## 元のコマンド

repo rootから実行した。以下の`RUN`は当時のoutputsディレクトリ。再実行時は保存済み証拠を上書きせず、
同じ`evaluate/`直下に別名のRUNを作る。`rescore.py`の以前のoutputs名は固定されている。
scriptは`__file__`から出力先を定めるため、bundleそのものを実行場所にしない。

```bash
RUN=outputs/tennis_scene/evaluate/court-reference-white-lines-20261116
BUNDLE=knowledge/runs/run-court-white-line-reference-20261116
mkdir -p "$RUN"
cp "$BUNDLE"/*.py "$BUNDLE/net-exclusions.json" "$RUN/"
PYTHONPATH=. .venv/bin/python "$RUN/capture_frames.py"
PYTHONPATH=. .venv/bin/python "$RUN/fit_reference.py" --output "$RUN/fit-a2"
PYTHONPATH=. .venv/bin/python "$RUN/balance_reference.py"
PYTHONPATH=. .venv/bin/python "$RUN/aggregate_reference.py"
PYTHONPATH=. .venv/bin/python "$RUN/rescore.py"
PYTHONPATH=. .venv/bin/python "$RUN/render_review.py"
PYTHONPATH=. .venv/bin/python "$RUN/render_remaining.py"
PYTHONPATH=. .venv/bin/python "$RUN/report_details.py"
```

`capture_frames.py`はdecodeされたBGRとPNGのhashを`frames.json`へ書く。デコーダ差はBGR hashで検出する。
`fit_reference.py`は48px→6pxの中間探索。ネットの帯は`net-exclusions.json`の座標を使い、モデル出力を読まない。
`balance_reference.py`は12px帯からframe単位の線対応を求める。全frameを必須にした中間referenceはcam3をnullとするが、
各frameの失敗と測定点は保存される。最終参照は`aggregate_reference.py`が明示的に2frame/clip以上・8frame/線以上の条件で作る。
cam3の失敗1frameは失敗のままで補完しない。最終の正本は`reference/references.json`だけ。

`aggregate_reference.py`は最終参照・入力manifest・net mask・fit script3本のhashを`reference/freeze.json`へ書く。
この時点以降に参照を変更せず`rescore.py`を実行した。`rescore.py`はreference hashと未承認フラグを確認してから、
旧/仮/共通可視maskの比較を保存する。今回のhashはbundleのfreezeと照合できる。

## 棄却した段階と保存場所

- `fit-a1`：ネット除外なしの試行でcam1の白いネット帯を誤対応。RGB確認で棄却し、予測を採点しなかった。
  そのsource snapshot・参照・棄却理由を保存した。当時はrootのfit scriptで実行したため、保存snapshotを
  `fit-a1/`内から直接実行するものではない。元の全測定はoutputsに保持。
- `fit-a2`：ネット除外後の中間H。全frameをpoolしたRANSACは早期のframeの変化も外れ値にしたため最終参照として不採用。
- `fit-final/camN-evidence.json`：frame別H・測定点・失敗・最終fitに使う間引き点。大きいためoutputsに保持し、hashは`source-identity.json`。
- `evaluation-secondary-snapshot`：後のscene集計を最初に読んだ結果。元の主報告との2選択の相違を検出して保存し、主表を正本snapshotで作り直した。
  閾値や推定を再選択したものではない。
- `reference/camN-measured-centres.csv`：最終参照の診断に使った中心点。frame列は各cameraの`frames.json`の順、line列はschemaの9線の順。
  大きなRGB・推定点配列・全画像はGitへ入れていない。

## 確認画像と集計

`review/index.html`は4camera×3clip×4領域の原解像度cropを原画/旧青/新橙/両方で切り替える。
`remaining/index.html`は仮FP全337件。元のframe0のdecode値を以前の推定入力hashと照合している。
Git内のoverview/代表図は一覧用に減色した。人間の再承認はoutputsの原解像度・原画切替で行う。
`report/line-motion.png`は同じ物理線のframe別残差中央値。cam1/3で時間変化があり、原因は未確定。

`evaluation/cases.csv`は276件×3条件×旧新参照。`summary.json`にcamera別の混同行列、正しい候補が存在する画像の救済率、
誤差、clipばらつき、共通mask比較、全候補の混同行列がある。`report/details.json`はラベル反転と参照差。
`framewise-sensitivity.csv`は同一frameの独立Hを得た19画像の補助診断だけで、主評価の参照を差し替えていない。

## 検証

```bash
PYTHONPATH=. .venv/bin/python -m pytest -o addopts='' --confcutdir="$RUN" "$RUN/test_reference.py" --durations=0
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py --check-summary --base-ref origin/main
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_repro_paths.py
.venv/bin/pre-commit run
```

数値テスト3件：白線の両edgeの中点、暗い影の棄却、線の外れ値除去→既知非affine Hの回復。
初回は最後のケースが実装の線外れ値除去を通さないloss単体検査になっており失敗した。
実際の処理契約へ直して許容誤差も0.8→0.1pxへ厳しくした。fit実装・凍結参照は変更していない。
初回と最終ログを保持する。src/testsの変更がないため、前回実施済みのCourt regressionを再実行していない。
CIはpush後の同じHEADを確認し、PRに結果を記録する。merge・issue closeは実施しない。
