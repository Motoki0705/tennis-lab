# #1559 実装前テスト設計

- 契約: 保存済みsigmoid mapを1frame/viewにつき1点へ復号する。scoreは元の格子peak値。
  既定はmainのargmax＋軸別log-parabolaと全配列が一致する。選択方式だけが座標を変える。
  局所fitが識別不能なら格子点を返し、理由コードを必ず返す。座標はW−1/H−1規約。
- 検出したい不具合: XY/時刻/view入替、交差項の欠落、logitをlogと誤る、端のパディングを観測と扱う、
  peak scoreの置換、既定出力の変化、設定だけ追加してpipelineで無視する、GT/LOVO除外viewの漏洩。
- 最軽量の層: 復号と補正はunit。中心既知の等方/軸別異方/回転/飽和/境界mapを非対称中心で生成し、
  独立した連続Gaussianまたはsigmoid二次曲面の中心と比較する。背景付き重心は誤差上限を明示する。
  設定・scene接続は既存component unitのstream/predictor境界だけをfakeにし、実moduleを実行する。
  replayは既存の非対称view fixtureとrunnerを使い、GTを渡すAPIを作らない。
- 再利用: tests/unit/tennis_scene/pipeline/components/test_ball_detection.py::test_predict_video_consumes_typed_task_prediction,
  ::test_overlapping_windows_select_all_evidence_atomically、同test_ball_evidence.py::test_artifact_roundtrip_is_exact_and_load_only_does_not_infer、
  tests/unit/tennis_scene/evaluation/ball3d/test_baseline.py::test_without_loess_keeps_acceptance_but_uses_original_detector_coordinates、
  同test_scoring.py::test_method_runner_excludes_labels_reference_and_heldout_camera。
  新規復号unitと、選択設定を実pipelineへ通す代表1caseを追加する。
- 予算: 新規40case以下、各5秒未満/setup1秒未満を目標。CPU・小mapのみ、checkpoint/動画/GPU不要。
  --collect-only と --durations=0 --durations-min=0、-n 0（最大2）で件数と実測を記録する。
  実clip全1010frameのreplay・動画は別のqualificationとして直列に実行し、最大RSSを測る。

比較計画: 全方式を同一固定bundleで全view/3 LOVO、LOESSあり/なしで実行する。
patch・温度・補正係数の選定は[0,505)のみ。選定結果をファイルへ固定してから[505,1010)を採点する。
前半推定の補正は前半のscoreableかつ検出peakが20px以内の対応だけによる頑健な軸別affineとし、
全cameraに共通の係数を方式設定として固定する（GT配列は方式に渡さない）。
半画素中心・stride4格子の理論補正は別条件として比較し、正しい教師規約と断定しない。
