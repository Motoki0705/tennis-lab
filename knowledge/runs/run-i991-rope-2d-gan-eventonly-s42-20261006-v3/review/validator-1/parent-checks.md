# 親の通常検証（原要件から独自に検査した後で参照）

- 関連pytest: 98 passed, 1 warning, 9.66s。座標2D/3D/Flow、Review API、旧v1 checkpoint、GAN境界・実更新、BLCS/PLCS共有callbackとDを対象。
- ruff/mypy: 合格（変更Python22ファイル）。commit hookでも合格。
- bash -n: 2本のqueue登録scriptに合格。
- GPU preflight: shared queue all枠、実dataset・batch32・実モデルサイズで4更新し、checkpoint/val/test/PNG保存まで終了。短い動作確認のためGANは1更新待機+2更新増加へ明示override。本学習の500+1000とは別。
- preflightの位置精度は学習品質の判定対象ではない。
- GPU preflight run: outputs/ball_refiner/train/rope-2d-gan-gpu-preflight/20261006-s42
- 原dataset hashは固定。CPU生成・学習・保存予測のroundtripをintegration testで照合。

GPU peak allocated: 1888051200 bytes
