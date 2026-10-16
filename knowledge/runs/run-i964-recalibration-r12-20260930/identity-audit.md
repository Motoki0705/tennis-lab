# 再利用時に見つかった外観maskの差（fit・新GPU結果の前）

run 12のCPU入力監査で、run 7のdev archiveの
`min_appearance_height_px=1` と、現在のproduction
`TrackingConfig.features.min_appearance_height_px=16` が異なると分かった。
run 11はproductionのtrack_sequenceへ保存特徴を直接渡していたため、
profile表示が16でも元archiveのmaskは1のまま入る。run 11の再現結果はその条件で有効。
新しいproductionの画像入口と完全に同じだったという根拠にはならない。

4 dev×3cameraの40,531 rowについて、productionと同じ丸め・画像端clip後の高さを調べると、
`video_000/clip_000/cam1` に1行、`video_001/clip_001/cam0` に8行、計9行が16px未満だった。
他の10cameraはこの差に該当しない。これはbbox/maskの監査だけで、人物ラベルや新しい精度は見ていない。

【要判断】既定を変更せず、過去の特徴をproductionの外観maskへ明示変換する
- 前提情報: detector/閾値/元row/ViTPose/CLIP checkpoint・cropは同一。
  旧cacheは余分な小cropのembeddingを持つ。9行の影響を精度上無害とはまだ判断していない。
- 選択肢: A: 16pxのproductionを維持し、旧cacheの該当行だけ外観valid=false・embedding=0にして
  CPUで該当cameraを再追跡する / B: productionを1pxに変更してrun 10のarchive条件に寄せる /
  C: dev特徴を全てGPU再計算する。
- 採用した暫定案と理由: **A**。今回の「tracker既定を変えない」を守り、9行に新しい推論も不要。
  変換は別artifactに出自/前後hashを残す。元cache/比較結果は保持する。
  新無ラベル特徴もproductionの16px規則で生成し、appearance batch8/pose batch4だけを資源設定として明示する。
  helperは実FeatureExtractorのmask処理との合成一致を検証する。
- 覆す場合の影響範囲: run 12の新特徴/較正の入力identityと、次runのCPU追跡。
  Bならproductionの仕様変更と全pipeline検証が必要。Cは追加GPU枠が必要。

この発見はprotocolのclip選定・擬似対応・fit・しきい値候補を変更しない。
**devは採点せず**、最終較正configのcommit後に初めて比較する。
旧run 11との差は、外観mask整合と較正の寄与を別に報告する。
その一回の比較バッチには、productionへ整合したtrack上の旧尺度と新尺度の2条件を使い、
run 11の数値は既存の参照値として併記する（同条件baselineとは称さない）。
10cameraの保存trackは入力値/設定/hashが一致する場合だけ再利用し、2cameraは再生成が必要。
batchの違いは値の同一性の証明とはせず、保存特徴を使う固定CPU比較であることも明記する。
