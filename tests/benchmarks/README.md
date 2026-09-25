# 実データの診断

`person_association.py`はPLCS Re-IDの実データを使い、fit/validation/test/checkpoint再読込を
小さな固定subsetで検証します。通常の単体テストには含めません。CUDA実行は共有training queueを使います。
モデル/データ契約と本学習recipeは[PLCS仕様](../../src/tasks/plcs/ASSOCIATION.md)を参照してください。

`coco17_placement.py`は身体配置の数値診断です。

`reid_checkpoint.py`は明示的に変換したRe-ID checkpointをtest splitで評価し、
元の保存embeddingと比較します。重み更新は行いません。CUDA実行は共有queue経由です。
