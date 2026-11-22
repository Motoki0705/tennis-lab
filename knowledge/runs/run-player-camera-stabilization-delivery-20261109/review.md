4秒比較の配信監査は成功し、強度選択と人間受入は未了です。

左上: 補正なし、右上: σ0.06秒、左下: σ0.08秒、右下: σ0.12秒。各240frame・59.94006fps・4.004秒、元frame [0,240)、各1280×720。

- [comparison_player_0.mp4](https://drive.google.com/file/d/1wCkX5_zLSewQUEh2_n4GiU8NvFHmnolV/view?usp=drivesdk) — `outputs/tennis_scene/visualize/player_view/stabilization-comparison/attempt2/full/videos/comparison_player_0.mp4`
- [comparison_player_1.mp4](https://drive.google.com/file/d/1JmTHLYsl_GhtnFpclFkWBBlOOQ03ZuCi/view?usp=drivesdk) — `outputs/tennis_scene/visualize/player_view/stabilization-comparison/attempt2/full/videos/comparison_player_1.mp4`

previewと配信sampleでは魚眼の湾曲、細線の段差/軟らかさ、ID1の大きな首振りによる灰色背景の広さを確認しました。対角159度は設計上の近似値です。動的なブレ感・臨場感は人間が判断します。

元のfull-a2はperiodic_save失敗/-15を保持し、全成果物のDrive内容照合と公式saveを復旧しました。詳細は対応run nodeと先行full-render runを参照してください。L4は14:18:29 UTCに停止、割当2時間55分39秒でした。

全1010frameは人間の強度選択後に描画します。人間の受入とC5の実GPU/背景support/メモリ/alias確認は残っています。
