# cam1-far重複boxへの次の提案（未実装・未実測）

【要判断】検出後・特徴生成/追跡前の明瞭な人物重複だけを抑制する
- 前提情報: [run 9の元row・座標・score](../run-i964-tracker-linking-r9-20260930/diagnosis-causes.json)では、
  clip_000/cam1 frame476のrow975/976が同じ遠側選手でIoU **0.822485**、score **0.431547/0.395793**。
  clip_013/cam1 frame176のrow354/355はIoU **0.528301**、score **0.415193/0.310963**。
  重複から競合IDが生まれた一方、clip_001には交差する別人へのID移行もある。
  全devは40,531検出/10,491 camera-frame（平均3.86box/frame）。
- 選択肢: A: 高IoUの人物boxに限定する決定的なclass-agnostic suppression /
  B: containmentも併用して入れ子boxまで消す /
  C: 現状を維持し、tracker内の重複ID整理だけを別途検討する。
  Aは単純で計算量が小さいが低IoU重複は残る。Bは小さい遠側boxを広く整理できる可能性がある一方、
  交差・遮蔽する別人の消去リスクが増す。Cは検出契約を保つが競合IDの発生原因を残す。
- 採用した暫定案と理由: **Aを次の承認対象として推薦するだけで、今回の実装・評価には入れない。**
  最初の候補仕様はIoU >= **0.80**でscoreの高いboxを保持し、完全同scoreなら元row昇順。
  全人物を対象に同一frameだけで行い、court選別や人物ラベルを参照しない。
  sourceのCOCO .30は維持し、別の明示的postprocess設定・cache identityにする。
  0.80はこの提案段階の仮値で、run 9の失敗を見た後の仮説である。事前独立な閾値とは主張せず、
  承認後に1条件として固定し、ラベル上の探索をしない。
- 覆す場合の影響範囲: detector後処理の公開設定、person_detections/features/tracksのcache identity、
  pipelineと#935共通人物処理、後続のqualification。既存run 8–10の結果は保持する。
  本runではdetector・閾値・NMSを変更していない。

期待効果（仮説）: 上のclip_000単frameならrow976の除外候補を一意に定められるため、
同じ人物を表す2trackの発生・競合や、その後の選別除外/断片化を減らせる可能性がある。
これは静的な不等式からの予想であり、全timelineのIDF1改善量は**未測定**。
clip_013のIoU .528重複はこの案で除けず、clip_001の別人交差も直接は解決しない。
交差中の本物2人がIoU .80を超えた場合は正検出を失うため、coverageが悪化する可能性もある。
scoreが高いboxの足元が常に正確とも限らない。

見積り（計測値ではない）:

|作業|見積り|成果/条件|
|---|---|---|
|共通後処理・設定/cache出自・単体テスト|エンジニア4–6時間|重複、別人交差、同score、空frame、元row保持、off時完全一致|
|既存dev cacheによる1条件のCPU比較と動画|CPU 10–20分を2段以内、実装/確認1–2時間|既存の検出/pose/CLIPの元rowを絞って使う。再推論/GPU不要。4thread以下、RAM 2GiB程度、追加disk 0.5GB以下を目安|
|本番接続後の全pipeline確認|別runでGPU枠の再見積りが必要|CPU cache比較はDINO→後処理→特徴生成のend-to-end保証ではない|

frame内の全対照合はO(N²)、devの平均約4boxでは小さい負荷が予想されるが、実測latencyはまだ無い。
次のdev比較では今回ユーザーが選ぶtrackerを固定して、重複からの新ID、cam1-farの保持とraw/group IDF1、
全camera/near-farの取り逃し、非選手残存、#933 pair F1/coverageを同時に見る。
改善量や採用は先取りしない。予約未見clipはそのdev比較でも使わず、全調整凍結後の1回に残す。
