RTMPose-X の入力を比較実装と同じ連続 NCHW 配置に修正しました。初回の PT16 照合では座標の最大差が 0.49658 px、score の最大差が 0.001343 で、事前基準を超えていました。再照合でも判定基準は同じです。

修正後の公式 MMPose との CPU 照合は、実画像の2 box・34関節で座標の最大差 0.0000324 px、score の最大差 0 で合格しました。関連テスト30件と pre-commit の ruff・mypy も合格しました。コミットは `851d3373b8ffbaf067fa8cc97a4f0fc249041e76` です。

初回の4カメラ実行は人物処理前のコート検出で停止しました。再実行では同じ4本の動画と照合済みの TensorRT エンジンを使い、採用済みの共有コート校正を指定しました。GPU の PT16・TRT16 照合と4カメラ実行を共有 queue に再投入しました。結果の確認と PR 作成が残っています。経緯は [#2664](https://github.com/Motoki0705/tennis-lab/issues/2664) に記録しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204262108023080348_2026-11-27T00-04-51-976919651Z_4034713_i2664-people-models-qualification-retry.job"]}}