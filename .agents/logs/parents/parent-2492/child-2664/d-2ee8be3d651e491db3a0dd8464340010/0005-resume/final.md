設定監査で検出された Python 側の実行時既定値を取り除き、設定ファイルから値を明示的に渡すよう修正しました。コミットは `1b419d7dda6a49844803286b56f3d9592644f6fd` です。

設定選択、既定経路のキャッシュ互換、TensorRT の識別に関する46件と、pre-commit の ruff・mypy が通りました。公式 MMPose との CPU 照合も、34関節で座標の最大差 0.0000324 px、score の最大差 0 で合格しました。

全体の設定一覧テストと設定監査を再投入しました。結果を確認後、push と PR 作成に進みます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261127T002243Z_2113499c475e_i2664-full-config-retry"]}}