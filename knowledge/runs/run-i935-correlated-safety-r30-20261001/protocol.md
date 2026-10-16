### 進捗ログ run 30 (2026-10-01) — 相関安全benchの実行前登録

**実行前に方法と合格規則を固定します。閾値の探索・再調整はしません。**

- 対象/判定: #932と同じsorted test[600:1000]の400scene、元の28条件、合計11,200scene-condition。元摂動RNG seed1。court_sideの全設定を元reportから固定しmargin=.15、ball-only。presence≥.9 AND 90%包含楕円面積≤30,000 source px²。
- 合格規則: **filteredのwrong decisionが全28条件を通じて0件のみPASS**。停止率と理由を全条件・pooledで報告する。FAILならclip_000 qualificationは進めず選択肢だけ提示し、自動採用しない。停止率に新しい合格閾値は追加しない。
- 残差: anchored_12k seed42の較正済みbank.npz SHA256 `0697fe921daf79c7960d616ed0437efecd3b7195f3f8a7dd6188048dd8ecc858`、calibration.json `2c9cd7ba9a1d63addeaecb808441fbae0447ad21e4a3a279ed1694fcbf67ae01`。observed条件のMeiji valだけを使いclip_000/video_001は除外する。**同じ実frame rowから全4成分残差・Cholesky・mix logits・presence logitsを一緒に採る**。信頼度だけの独立シャッフルは行わない。
- 時間: residual RNG seed30001。empirical cameraを最初の3viewへpermutationし、4台目は明示的な独立camera抽出。各viewで最大30 synthetic-frame block、Meiji 59.94fpsからstride2。artifact境界/欠測/continues=falseではblockを切り、別blockを抽出する。frameごとのrow出自を追跡可能にする。
- 28条件の保持: 元の校正・同期ずれ・欠測・画素noise・static false distractor摂動とRNGをそのまま保持。その合成点に、上の**相関した実GMM残差全成分**を正規化座標で足し、平均を[0,1]へclip。共分散は再fitせずsource画素へ変換し、productionのpoint_confidenceで面積と最大weight点を計算。未選別/選別を全く同じGMMで対比較する。元bench単体の全条件集計一致も検査する。
- 限界: empirical残差とconfidenceの対応は厳密に保持するが、追加された静的偽点/同期/校正のstressは実refinerが識別したという仮定を置かず、confidentな偽点も残る。全誤差を実refinerのE2E分布だとは主張しない。実測部分と追加stressを分けて、選別前後の実際の点誤差も報告する。元のJPEG入力bankの外的妥当性・mp4との差は残る。
- run29診断: 同じseed/連続confidence blockを再現し、3wrong sceneだけの選択camera、source/false点、distinct frame、camera pair支持数、全仮説support/costとmarginを保存する。旧失敗記録は上書きしない。
- 次の一手: この事前規則の実装テスト・実行・結果記録。GPUを使わずCPU最大4thread。
