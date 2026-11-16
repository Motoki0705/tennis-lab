# 修正前の設計・検証契約

対象は #2190 / PR #2200、base fa7bb8ebe。親note #2150 issuecomment-6925940896 による再設計。GPU、queue、infra操作は行わない。run73のraw、LINE場、元RGBを共有してCPUだけで比較する。承認済み参照と10/25px、FP0・正しい推定採用90%は固定する。

- (a): 各最適化の支持集合を固定する。不整合な点を含む解を黙って別Hへ置換しない。明示した候補の生成・親子関係・実際の集合・残差・停止理由を残す。空間分布とconfidenceの候補を保持する。境界の片側観測と重複の相関群は維持する。
- RGB: coarse stripeがあるだけでは必須線としない。初期モデル投影との整合、他の物理線との整合、代替stripeを確認して必須集合を決める。疑わしいstripeの喪失は診断のみ。確かな線の喪失、曖昧な対応、支持外の不確かさの拒否は残す。
- (a)品質: 最終Hを制約した観測と、公開する可視投影に対する感度を点検する。都合のよい領域だけで精度を報告しない。

実装前に検討するテスト:

|契約・検出する故障|再利用node/層|追加検証・コスト|
|---|---|---|
|最終Hと固定集合が一致。1点の脱落で他の分布した支持を失わず、循環/旧H代用なし|geometry/test_hybrid_homography.py::test_line_corrects_biased_kp_and_excludes_confident_outliers / test_nonconvergent_joint_fit_has_no_kp_only_fallback（unit）|既知H＋残差境界の点を追加。保存rawのsynthetic58件で原因と回復/未回復を追跡|
|可視投影の不確かさ、退化と片側観測の区別|geometry/test_quality.py、test_observations.py（unit）|画面外の遠方点だけが不安定な場合と、可視点まで不安定な場合。手計算・独立投影に照合|
|stripeと塗線の対応を区別。誤ったネット/影を必須にしないが、確認済み線を捨てない|geometry/test_paint_refinement.py::test_rgb_centres_recover_known_h_with_net_and_occlusion / test_extended_wrong_tape_cannot_be_silently_discarded（unit）|既知Hで描く塗線・余分な平行stripe・欠線・遮蔽物。旧テストの『誤stripeも必須』契約は今回の指示に合わせて置換し、誤対応を採用しない契約を残す|
|RGB失敗時の旧H流用禁止、診断/設定/cacheの一致|inference/test_regions.py、tennis_scene/test_pipeline_configuration.py（既存unit）|schemaと最終/初期Hの分離を拡張|

各unit caseは5秒以内を目安に --durations=0 で測定（未実測）。重い全件比較は受入記録へ分離する。新しいGPU・注釈・rawは取得しない。

設計には Court val、synthetic、既知Hの人工画像を使う。meiji_3camの減少原因を診断するが、meiji_4camはソース・全設定を凍結した後の最後の評価だけに使う。旧処理・前回(a)・前回(a+b)はrun73の結果を再利用し、新(a)と新(a+b)を追加して比較する。要求の4条件表に加え、新(a)単独の表を残して修復効果を分離する。結果後に閾値を変更しない。

## 非meiji4の設計で確かめた問題と変更

1. seed時の固定集合が最終LINE fitで不整合になる。固定集合ごとの独立したsolveを維持し、最適化後に発見した空間/信頼度集合を親candidate IDつきでenqueueする。同じ集合は同じseedで二度解かず、最大8候補。閾値の変更はない。異なる集合の解を同じsolveと偽らない。成立した解も保持し、順位の固定点を要求しない。
2. 全coarse stripeを同じ重みで当てはめると、隣のサイドラインに対応をずらしてもコート幅を変えたHが成立する（既知H人工画像で検出）。モデル近傍のstripeとcoarse stripeを候補として2方向2本ずつのgridを列挙し、残る物理線が検証する。確認された近傍の対応数、全体の支持数、残差の順に選ぶ。未確認の遠いネットstripeの喪失は拒否せず、確認済み観測の喪失/位置変更は拒否する。
3. ある方向が2本のとき、1本のleave-outはrank不足になる。有限差分の微小誤差をrank8と誤認すると、任意のHへの数百pxの移動を『影響』と数えてしまう。既存condition上限でrankを検査し、計算不能はessential_linesへ分離する。不可欠な線の意味確認と、全線を用いた投影分散を必須にする。既知Hの2 baseline+3 sidelineは合格、近側に偏った5線は外挿分散で不合格であることをテストする。
4. 数値の許容距離・誤差・coverage・variance上限は変更しない。6pxのモデル近傍帯、3pxの線残差、condition10000等は既存設定を用途ごとに明示する。追加は候補数の計算上限8だけ。
