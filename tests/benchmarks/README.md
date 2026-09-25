# 実データの診断

`person_association.py`はPLCS Re-IDの実データを使い、fit/validation/test/checkpoint再読込を
小さな固定subsetで検証します。通常の単体テストには含めません。CUDA実行は共有training queueを使います。
モデル/データ契約と本学習recipeは[PLCS仕様](../../src/tasks/plcs/ASSOCIATION.md)を参照してください。

`coco17_placement.py`は身体配置の数値診断です。

`reid_checkpoint.py`は明示的に変換したRe-ID checkpointをtest splitで評価し、
元の保存embeddingと比較します。重み更新は行いません。CUDA実行は共有queue経由です。

`component_pipeline.py`はclip動画から宣言型pipelineを検証し、外部ball注釈と幾何確認済みsideを明示loadします。学習済みRe-IDの生embedding・予測IDを別artifactに保存してから、既存の人手人物対応をbbox時系列で現trackへ照合し、検証用に同schemaの確認済みartifactをloadします。対象外の人物trackは元の検出・追跡成果物に残し、以後のplayer軸から除外します。GPU実行は共有training queue経由です。各stageの実行/再利用、出自、全frameのvalidity、scene export、全段load-only再開を記録します。確認済みIDによる下流検証をモデルRe-IDの正解率と混同しません。
