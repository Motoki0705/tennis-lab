# 続き（同じsession・branch・worktree）: まとめてPRを出す

固定shapeの4件（`i2335-r5-gvhmr-static-{b2,b4}-{l145,l177}`）は、4件とも `gvhmr build-2: exit -11, verified=False`（SIGSEGV）で失敗した。

**親の判断（前回の指示どおり）**: これ以上は追わない。GVHMR TRT FP16 のB>1は「ビルド不可」として、次を根拠にrunノード・metrics・validationへ記録する。
- 試した設定: 動的バッチprofile、builder最適化レベル0、B=2・4の固定shapeを別エンジンで作成、L=145・177。
- 各試行の終了状態。
- verboseログで最後に記録されたTiming Runnerの層（`ForeignNode[[local path]]` など）。

B=1のTRTは #2305 で動くので、パイプラインで人物を束ねずに推論する場合の扱いにも触れる。GPUジョブは新たに投入しない。

**作業**: verboseログの要点（最後の層・ScatterNDまわり）を記録する。そのうえで、`Closes #2335` の日本語ドラフトPRを作ってpushし、`PARENT_RESULT` のCI待ちを宣言して終了する（`resume_on_success: true`、next_step: ドラフト解除と最終報告）。

**最終報告に入れること**:
- モデル×条件×バッチの要約表
- 代表バッチでのncu指標と、batch 1との比較
- #2305 の旧PTQグラフの検査結果
- GVHMRのバッチ化が不可である根拠
- #2336（ViTPose-H・HMR2・DINO）へ引き継ぐharnessの使い方
