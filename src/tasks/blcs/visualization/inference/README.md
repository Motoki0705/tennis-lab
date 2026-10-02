# BLCS Inference UI

起動コマンドと操作は[Web UIガイド](../README.md)を参照してください。
対象は`blcs/single_object`のphysical_v1シーンと`blcs_multiview_axial`のcheckpointです。

`service.py`はシーンとcheckpointの一覧、要求検証、推論結果を担当します。
checkpointのモデル・座標契約を確認し、複数カメラの対象1つを推論します。
track-query、multi_object、reference modelは選択できません。
GPU推論は共有training queue経由で実行します。
