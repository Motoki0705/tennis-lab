記録専用の [PR #2698](https://github.com/Motoki0705/tennis-lab/pull/2698) を ready で作成し、CI の開始を確認しました。ブランチは `codex/parent-2627-issue-2696-primary`、PR HEAD は `f3ba146f57c07d8f1e168cb03fc6e71b9b9fdac1` です。

固定3クリップを同一HEAD `2dbd8c355faa2d530b8b0e17164f691e60a57d07` で各条件1回実行し、全12シーンを生成しました。CLI 時間は C1〜C4 の順に **2,186.31、1,686.15、1,706.81、1,319.31秒**でした。モデル共有で500.17秒短縮し、横断バッチの追加で20.66秒増加しました。TensorRT FP16 は C3 より387.50秒短く、Peak VRAM は4.49→1.76 GiB、Peak RSS は8.16→3.77 GiBでした。 [計測記録と比較図](https://github.com/Motoki0705/tennis-lab/tree/f3ba146f57c07d8f1e168cb03fc6e71b9b9fdac1/knowledge/runs/run-i2696-exec-system-ablation)

全条件で C1 と12選手IDが対応しました。3D関節座標は C2 で完全一致し、C3 の最大差は0.079 mmでした。C4 は **P95 9.06 mm、最大0.698 m**で、SMPL有効選手フレームは3,496→3,495でした。C1〜C3 の CLIP-ReID・HMR2・GVHMR には forward 限定の FP16 autocast を適用し、GVHMR のクリップ内バッチ上限は全条件で4に揃えました。

JSON・CSV・PNG・PDF の6ファイルは再生成でバイト一致し、pytest 56件と ruff が通過しました。`kg_validate` はベースにある RF-DETR-L・RTMPose-X の deployment 未登録2件を報告し、今回のノード・run・summary 由来のエラーは0件でした。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}