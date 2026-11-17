# 共通コートのdrift診断

[English](README.md) · [简体中文](README.zh-CN.md)

`inspect_frame(rgb, shared_h, supported_lines, DriftConfig())` は、共有fitで確定したHと線の意味対応を使い、1 frameを独立に診断する。線の対応・端点・ネット帯を再推定しない。既存のエッジ対抽出で白線中心と競合する白帯を測り、既知の線に対する符号付き垂直offsetから平行移動・回転・一様scaleの4自由度を求める。

交わる2線だけでは交点回りのscaleが決まらないため、初期解は3線から作る。採用には4線以上、横線1本以上・縦線2本以上、12度以上離れた2方向とrank 4が必要。矛盾した線、異なる同等支持の変位、rank不足、不確かさ超過は `drift_unobservable` として失敗する。自由度を落とす／局所Hへ戻るフォールバックはない。

1物理線につき10/90 percentileの2アンカーだけを数え、画素数を独立支持数にしない。アンカーのノイズ下限1pxと残差から共分散を求め、可視KPへ伝播する。投影std上限5px、線残差上限3px、探索半径48px。いずれも1920×1080の対角長からscaleする。

変位meanから2stdを引いて3px超なら `camera_moved_possible` のwarning。信頼できるmeanが10px超またはmaxが25px超なら失敗。検査不能も失敗。変位は診断用であり、下流へ渡すコートは共通Hのまま。

`summarize_clip` は全frameを保持し、失敗・warning・okの順で最悪frameを選ぶ。時間平均や同じ姿勢の仮定は使わない。未測定frameは保証できない。4自由度で表せない実際のprojectiveな動きも矛盾として失敗しうるので、失敗だけからcameraの移動有無は断定しない。

最低支持と不確かさは人工開発集合で凍結した。[run 78](../../../../../knowledge/nodes/court_detection/000078-run-court-video-drift-lines-20261117.md)に独立評価、frame選択と限界を残す。保存・明示loadは[動画成果物の境界](../../../../tennis_scene/pipeline/video_court/README.ja.md)を参照。
