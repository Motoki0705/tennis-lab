RF-DETR-L と RTMPose-X を追加し、検出と pose のモデル・backend を個別に選択できるようにしました。既定は DINO＋ViTPose-H の PyTorch FP32 です。[PR #2681](https://github.com/Motoki0705/tennis-lab/pull/2681)

- 最終 HEAD `46c25d1b499f3f6367bcae8b6fbd758643ce3b3f` で pytest 757件と設定監査140境界が合格しました。出力先テスト116件は共有ストレージから隔離して実行しました。公式 MMPose との CPU 照合も基準内でした。
- GPU の PT16・TRT16 照合は全4条件で座標・score の最大差が 0 でした。4カメラ・369フレームの標準シーン実行は48段すべて完了しました。最終 HEAD でモデル、実行設定、重み、TensorRT plan が GPU 実行時の成果と一致することも確認しました。[実行記録](https://github.com/Motoki0705/tennis-lab/issues/2664#issuecomment-7206315590)

#2470 の3D追跡から公開までの通し確認と、代表6クリップの品質比較は #2487 に引き継ぎます。PR の [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45918392208) は実行中です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2681,"head":"46c25d1b499f3f6367bcae8b6fbd758643ce3b3f","resume_on_success":false}}