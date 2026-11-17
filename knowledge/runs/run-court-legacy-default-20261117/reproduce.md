# 旧処理との完全一致replay

このrunはrevision 4の互換性受入。参照やGTへの精度を上げる実験ではない。
`baseline-source.json`に記録したPR分岐元の旧コードと、`protocol.json`の変更後コードを
同じPython/NumPy/SciPy/OpenCVプロセスで比較する。
実装commitは`evaluated-source.json`に記録し、実行開始時の内容と一致を検査した。元のhybrid/LINE/confidence solverはrun 70の
保存ソースとbyte一致することを確認した。region選択は旧ファイルの無変更コピーを使用する。

全3545画像/8530領域で両solverを実行し、matrix・投影点・最終支持mask・残差は
shape/dtype/bytes一致、status・candidate/refined件数・代替score gapは値一致を確認する。
public region APIへ両結果を渡し、全候補の採否と旧診断フィールド、選択regionも一致を確認する。
先頭の全画面領域は領域探索なしの出力も比較する。空の採用Hは双方Noneで一致させ、
失敗時のdiagnostic Hを採用結果として数えない。Court val/syntheticは保存済みfull-frame領域だけ、
sceneは保存された全proposalを比較する。YAMLとdataclassの既定の一致も実行時に検査する。

入力はrun 73のmanifest・RGB・保存LINE配列と、run 70のraw。各LINE配列が存在しなければ停止し、
再抽出・ニューラル推論はしない。native KP heatmap / LINE logitsは保存されておらず、
この境界で使わないためtyped wrapperの対応fieldは空tensorと明示する。
GPUは使わない。shared lockを操作せず、queueにも登録しない。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python knowledge/runs/run-court-legacy-default-20261117/check_inputs.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python knowledge/runs/run-court-legacy-default-20261117/replay.py --workers 10
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python knowledge/runs/run-court-legacy-default-20261117/check_mutations.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python knowledge/runs/run-court-legacy-default-20261117/archive.py
```

`protocol.json`は実行開始時の内容hashを記録し、終了時にも不変を検査する。
数値ソースのcommitと再現bundleの保存commitは異なる。`repro.commit` /
`evaluated-source.json`は数値ソースを指す。再現にはこのbundle保存版のscriptを使い、
数値ファイルが`protocol.json`のhashと一致するcheckoutで実行する。
保存結果はoutputsの`evaluation/<key>/comparison.json`。全旧/新Hと両候補診断を保存する。
Gitには小さい集計・再現script・ログ・inventoryの場所とhashだけを残す。
再実行は同じoutputへ書くため、別計算環境で比較する場合はoutput先を別にして旧結果を保存する。

過去run 70のsynthetic旧処理は573件成立、run 73の保存LINEを使った旧replayは579件成立だった。
初回のLINE距離場は保存されておらず、その数値差を今回の実装差と混同しない。
今回の根拠は入力を揃えた旧ソースとの完全一致であり、未保存の初回距離場・別数値ライブラリ・
CUDA forwardでのbit一致を主張しない。rawと参照は変更しない。
