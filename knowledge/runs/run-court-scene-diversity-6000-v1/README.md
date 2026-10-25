# 固定6000枚によるscene数比較の生成bundle

#983の入力snapshotに記録されたB00〜B03のCourt storeをdescriptor/indexでpinし、
そのmetadataの既存train軌道だけを密に再描画する。元datasetと3DGS資産は変更しない。
今回の固定枚数実験は#1343の全採択データ実験とは独立である。

| 条件 | B00 train/val | B01 train/val | B02 train/val | B03 train/val |
|---|---:|---:|---:|---:|
| scene-1 | 5500/500 | — | — | — |
| scene-2 | 2750/250 | 2750/250 | — | — |
| scene-3 | 1833/167 | 1833/167 | 1834/166 | — |
| scene-4 | 1375/125 | 1375/125 | 1375/125 | 1375/125 |

各条件は合成train5500枚、合成val500枚。実画像val2211枚は学習担当の別評価である。
元synthetic validation/testの軌道はどちらにも使用しない。

## 固定するもの

- 軌道・中心・形状・orientation・vertical phase・選択済み画角・注視高・解像度は
  公開metadataを権威とする。B01の画角45/67.5/90度も保持する。
- 元の4096区間arc table、最小24点、8点単位、最大1.05mを再現し、
  B00/B01をdensity2、B02/B03をdensity1へ細分する。全旧train画像のcamera位置とKを照合する。
- 新しい点のjitterはgeneration seed42、stable point ID由来のPCG64。
  元と同じ半径1mの円盤面積一様分布で、旧sample index由来の乱数列を再利用しない。
  density変更で共通点のID・camera・教師が変化しないことをCPUテストする。
- QAは旧条件のgeometry all-court KP>=4、renderer visible KP>=1。
  target KP・ROI穴率等を診断として記録し、#1343のD5を自動適用しない。
- sceneごとの元train軌道をseed42のstable hashで並べ、先頭ceil(10%)をval用として
  描画前に固定する。同じ軌道が条件間でtrain/valに移動しない。
- QA採択画像をtrain/val軌道へ分離後、それぞれのpool内で採択数に比例する
  virtual finish timeにより軌道を選び、軌道内では円周上のarc fractionのmaximin順で選ぶ。
  同率はstable hashで解決し、入力順・描画完了順に依存しない。小条件は同じ順位のprefix。
  複製画像で枚数を増やさない。不足した場合は完了として公開しない。

## 成果物

出力rootは`data/court_scene_diversity_6000_v1`。
`plan-audit.json`が入力と固定requestを示し、`trajectory-partition.json`が元train軌道の
train/val割当を示す。`generation-completion.json`がQA済み全poolの
`court_scale_dataset_version_v1` manifestを示す。
poolは保存契約としてtrainのみを保持する。

`scene-{1,2,3,4}.{train,val}.subset.json`の8ファイルは
`court_dataset_subset_v1`。キーは`schema, source_manifest,
source_manifest_sha256, selection_seed, split, samples`。
`samples`は`{scene_id,sample_id}`の順序付きリストで、sample IDにscene prefixを付けない。
source manifestは同じroot直下の全pool版への相対参照。
subsetの`split=val`は学習側でremapし、共有poolの疎recordを書き換えない。

## 再現

メイン`.venv`を利用する専用worktreeで、まず本bundleの生成実装commitをcheckoutする。
`REPO`は元資産のroot、`OUT`は新しい出力先とする。

```bash
REPO=/home/kamimura/projects/tennis-lab
OUT="$REPO/data/court_scene_diversity_6000_v1"
SCRIPT=knowledge/runs/run-court-scene-diversity-6000-v1/generate.py
PYTHONPATH=. .venv/bin/python "$SCRIPT" plan --repo-root "$REPO" --output-root "$OUT"
```

GPUのsmoke/renderは共有training queueの`resource=all`で実行する。
smokeは各scene8視点のRGB・教師overlayを保存する。人間または担当エージェントが画像を確認してから
renderを別jobとして登録する。どちらもqueueの`TENNIS_RUN_ID`が必要である。

```bash
PYTHONPATH=. .venv/bin/python "$SCRIPT" smoke --output-root "$OUT"
PYTHONPATH=. .venv/bin/python "$SCRIPT" render --output-root "$OUT"
PYTHONPATH=. .venv/bin/python "$SCRIPT" finalize --output-root "$OUT"
```

最後のfinalizeはCPUで全JPEG・疎教師・QA ledgerを厳格再検査し、8条件を公開する。
`selection-report.json`に実数・unique ID/camera/JPEG・target KP/coverage/ROI診断を記録する。
`review/`にscene別のRGB/教師contact sheetと連続軌道動画を保存する。

通常検証は`tests/unit/synthetic_data_generation/dataset/court/test_saved_train_scale.py`、
既存scale planning/publication、および生成CLIのテスト。
