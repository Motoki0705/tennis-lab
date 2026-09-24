# 実データの診断

`person_association.py`はPLCS Re-IDの実データを使い、fit/validation/test/checkpoint再読込を
小さな固定subsetで検証します。通常の単体テストには含めません。CUDA実行は共有training queueを使います。
モデル/データ契約と本学習recipeは[PLCS仕様](../../src/tasks/plcs/ASSOCIATION.md)を参照してください。

`coco17_placement.py`は身体配置の数値診断です。
