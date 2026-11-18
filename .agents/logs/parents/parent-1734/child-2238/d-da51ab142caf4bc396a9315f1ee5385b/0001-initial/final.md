[PR #2240](https://github.com/Motoki0705/tennis-lab/pull/2240) を **Ready for review** にしました。ブランチは `codex/person-stage-speed`、HEADは `38b83a8b4` です。

DINO・ViTPoseの精度を `fp32`、`bf16`、`fp16` から選べるようにし、cache identityにも反映しました。既定はFP32です。DINOの変形可能アテンションは混合精度でもFP32で実行します。既定FP32の検出と姿勢は、固定8画像とclip_000の全4カメラ・356フレームで旧出力と完全一致しました。

- 全尺12ケースで、DINOはBF16・FP16ともに約1.72倍、ViTPoseはBF16で2.65倍、FP16で2.79倍、flipなしで1.97倍でした。ViTPose FP16では追跡・選別・人物対応・3D有効maskが一致し、clip_000の3D関節差はP95が0.1 mm未満、最大30.9 cmでした。
- 姿勢なし追跡のCPU再生では、選別後の姿勢推定対象が約52〜53%減る試算でした。clip_000では選手選別と人物ID割当が変わりました。この構成の推論時間と3D品質は未計測です。
- clip_037の人物対応と3Dは、通常のcourt side判定が停止したため、事前に固定した診断用マージン0.12で比較しました。共有キューのTERMによる中断後は、完了済み5ケースを保持して残りを実行しました。 [全尺計測・VRAM・重ね描き5点](https://github.com/Motoki0705/tennis-lab/blob/38b83a8b452c83f7ee581791c2d122864e11ceb4/knowledge/nodes/tennis_scene/000050-run-person-stage-precision-meiji-20261118.md)

関連CPUテスト1,468件とpre-commitが通過しました。初回CIの注釈E2Eは同じHEADでの再実行で通過し、`Python` と `Knowledge library` は成功しました。ローカルの注釈E2Eは、PR・mainとも共有venvの `os.pidfd_open` 不足で停止しました。独立validatorは未指定で、試行・完了とも0回です。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44239875199/attempts/2)・[Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44239875121)