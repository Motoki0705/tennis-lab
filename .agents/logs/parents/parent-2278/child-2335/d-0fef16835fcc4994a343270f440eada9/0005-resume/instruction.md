# 続き（同じsession・branch・worktree）

前回のターンで投入した2件（`i2335-r4-gvhmr-builder0-l145`・`-l177`）は、どちらも終了値1で失敗した。ログには `gvhmr build-2: exit -11, verified=False` とある。ビルダー最適化レベル0でもSIGSEGVが起きている。

親の判断:
1. **GVHMR TRT FP16のバッチ化は、あと1回だけ方式を変えて試す**
   - バッチ次元を動的profileにせず、B=2とB=4をそれぞれ固定shape（L固定）の別エンジンとしてビルドする。
   - B=1の固定エンジンは #2305 で成功している。
   - ビルド時のログ（TRTのverbose・失敗した層の名前）を残す。
   - それでもSIGSEGVになったら、それ以上は追わない。「GVHMR TRT FP16のB>1はビルド不可」として、試した設定（動的profile・opt level 0・固定shape）と終了状態を根拠に記録し、まとめに進む。
2. **ncuのkernel数の不一致**: ncuがnsysとカーネル数で一致しなかった2件について、どの条件・バッチか、原因は何かを記録する。正式な値として使えないなら、その旨を明記する。
3. **GVHMRの旧PTQグラフの検査（CPUのみ、GPUジョブは不要）**: #2304 のレビューで、旧PTQ設定（#2297）のDINOのINT8・FP8グラフでは、指定していない演算（Add・LayerNormalizationなど）にもDQ出力が直接渡っていた。マージ済みの #2305 の `knowledge/runs/run-i2305-gvhmr-tensorrt/` で、実測に使った3a・3bの量子化ONNXに、main の `verify_qdq_scope` 相当の検査をかける。指定外の演算への接続の有無と、演算種別ごとの件数を、この子のrunノードに記録する（#2305 のファイルは書き換えず、記録だけをこちらに置く）。
4. **まとめ**
   - knowledgeのrunノードを更新し、`Closes #2335` の日本語ドラフトPRを作ってpushする。
   - origin/main（#2342 まで入っている）を取り込み、共有harnessのテストを通す。

GPUジョブを投入したら、`PARENT_RESULT` のqueue待ちを宣言して終了する。PRをpushしたら、CI待ちを宣言して終了する（`resume_on_success: true`、next_step: ドラフト解除と最終報告）。

最終報告には次の2点を入れる。
- モデル×条件×バッチの要約表と、代表バッチでのncu指標のbatch 1との比較
- #2336（ViTPose-H・HMR2・DINO）へ引き継ぐharnessの使い方
