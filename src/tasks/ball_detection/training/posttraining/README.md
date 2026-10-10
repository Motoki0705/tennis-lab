# Robust query posttraining / CNN campaign

先行研究・採用仮説の正本は [遮蔽/カメラ拡張 #1049](https://github.com/Motoki0705/tennis-lab/issues/1049) と
[CNN比較 #1050](https://github.com/Motoki0705/tennis-lab/issues/1050)。ここでは実装契約と実行を記す。

## 構成とデータ

`models/mdd_pretrain/config.py` の `encoder_variant` は `residual` / `convnext_v2` / `fasternet`、
`temporal_mixing` は `dense3d` / `factorized`。baselineを維持し、追加2設定はfactorizedを使用する。
ConvNeXt V2系は7×7 depthwise、4倍pointwise、channel LayerNorm、frame独立GRN。
FasterNet系は1/4チャネルのPConv、2倍pointwise、frame独立GroupNorm。
factorizedは中間幅Cの空間3×3→時間3×1×1で、パラメータ数を揃える原論文設定とは異なる。
CNN出力の4解像度・チャンネル・時間参照範囲を揃え、同じDPTを使う。
FP32 GRN energyはreductionで蓄積し、大きなFP32二乗feature mapを作らない。

pretraining v2 checkpointはCNN種別を明示。過去v1は残差CNN＋dense3dとしてのみ明示的に読み、
異種CNNへの転送・MDD/decoder/manifest不一致を拒否する。
事前学習は全候補とも拡張なし。現在のbaseline runを継続して比較に使う。

## 事後学習の拡張

`augmentation.py` はGPU RGB uint8に処理し、model内でMDDを再生成する。RGB特徴の別入力経路はない。
パラメータの正本は `configs/augmentation/mdd_posttraining.yaml`。

- **camera**: real PTSに沿うpan・zoom・小さいroll、滑らかなsinusoidal jitter。GTにも同一行列。
  画像は逆行列でsample、座標は順行列で変換する。pixelのアスペクト比を保つ回転。
  画面外GTは座標lossから外す。観測点の25%以上を失う、または8点未満になる変換は
  clip全体でidentityに戻し、`camera_rejected`を記録する。境界はborder padding。
- **occlusion**: 2〜6枚の連続区間へclip内一定色の移動矩形を合成する。半数はball近傍、
  半数は任意位置。ball-near矩形もanchorにoffsetを加え、その後の速度はGTと独立にする。
  GT軌跡を追いかける矩形を作らない。全32枚を隠さず、人工遮蔽した既知GTは保持する。
  元から不明な位置へ教師を作らない。マスクで覆われた点を`artificially_occluded`に記録。
- seed/epoch/clip/start/frame_step/profileのhashから乱数を決め、worker順に依存しない。
  clean validationでは拡張しない。stressはcamera/occlusion/combinedを固定seedで別評価する。

`runner.py` は最初の1 epochでCNNを固定、後続epochでCNNをdecoderの0.1倍LRで解凍する。
SwiGLU中間幅704、dim256、4層、8heads。同時刻cross→query時間self→FFN。
全10 epoch×6000窓、AdamW、decoder peak LR1e-4、warmup500後cosineで1/10、BF16。
正解位置はobserved-only SmoothL1。既存JPEG decode・prefetch・窓samplingを再利用する。

clean common validationのFPS等重み平均位置誤差でcheckpointを選択し、testを使わない。
stressはclean best選択後だけ実行し、人工遮蔽subsetも同じframe所有者から集計する。
各epochのGIF、train時の変換行列・mask、段階別LR、checkpoint、再開情報を保存する。

## 実行

すべてのCUDA commandは元repoのshared training queueで直列実行する。

1. `train_cnn_candidate.py --prefetch-mode overlap|serial`: GPU入力の実行方式を明示する。
   `overlap`は720p・96更新のGPU smoke成功後、scratchから6万更新。
   `serial`はnvJPEGを維持して並列画像先読みを停止し、full manifest・10epochのLRスケジュールのまま
   `--stop-after-epoch 0`で最初の6000更新＋validation・checkpoint保存後にprocessを終了する。
   保存hash・source・data・finite重み/勾配を検証し、別processで同じcheckpointから残り54000更新を続ける。
   短縮診断を本学習完了と扱わず、診断失敗時は再開しない。
2. `advance_cnn_campaign.py --plan /absolute/campaign/plan.json`: 2候補を重複なくqueueへ投入。
   3runのCOMPLETED・checkpoint hash・data/budget/seed/precision/DPT一致を確認して比較する。
   common誤差、同値ならfull誤差、その次に名前順でbestを1つ選択。
3. bestだけ `train_best_mdd_posttraining.py` へ投入。別processの
   `probe_mdd_posttraining.py` でGPU上の拡張、compile後のfreeze→unfreeze、CNN勾配・重み更新を確認してから
   `posttrain_mdd_query.py` を新processで実行する。probeの重み/optimizerは本学習へ引き継がない。
4. 比較の表/JSON/PDF、best事後学習のclean/stress metrics・GIF・checkpointを保存する。

planは`code_root`（固定worktree）、`queue_directory`、`manifest`、`baseline_run`、`thread_id`、
`training_root`、`smoke_root`、`posttraining_run`を持つ。学習はoutputs/ball_detection/train配下、
GPU probeはanalyze配下、campaign rootには比較成果物と制御情報を配置する。
v2 planはさらに各候補の`run_id`/`job_name`/`smoke_id`/`prefetch_mode`と、
`posttraining_prefetch_mode`を明示する。v1は元のrun名・overlapへ明示的に展開される。
v3はさらに`cuda_launch_blocking`を明示し、新規queue payload（事後学習probeを含む）へ環境変数を渡す。
v1/v2はFalseへ明示移行する。復旧時の起動環境変更は各runの`resume_receipts/`に保存し、比較へ含める。
既存checkpointを使う診断は`tests/benchmarks/ball_dpt_checkpoint_replay.py`で、失敗batchを含む少数更新を
同期実行する。元checkpoint・本学習出力は変更せず、診断重みも本学習へ引き継がない。
再試行は失敗出力を保持し、新しいrun/job名を指定する。他の実行中jobは名前で再利用する。
`advance`はファイルlockとqueue名の照合で二重投入を防ぎ、失敗したjobを成功とみなさない。
失敗は監視側で調査してから、既存checkpointでの再開または新しいrun-idを判断する。

今回の運用はPR #1047版CLI heartbeat（120分間隔）。実行中は一度確認して静かに終了し、
全pretraining終了後にbestを選び事後学習へ進む。全成果を確認した後にtimerをpauseする。
CLIとマシンの稼働が必要。単にtimerがactiveであることと、実際に継続ターンが完了したことを区別する。

## 確認用

- `profile_mdd_cnn.py --output ...`: CPU/metaのparameter数とConv/Linear MACs。正規化・メモリ転送・backwardを含まず、実速度ではない。
- `preview_posttraining_augmentation.py`: train clipをCPU/OpenCVで可視化。`--targeted-demo`はレビュー用にball-nearを強制し、traceへ明記する。学習のdecoderはnvJPEGのまま。
- CPUテスト: 同期座標、画面外、遮蔽教師、clean非拡張、seed再現性、時間参照範囲、v1転送、freeze/解凍、比較/queueの受入条件。

拡張の2D affineは実カメラの視差・新視点やscene cutを再現しない。論文由来のCNNや拡張の効果は
このデータで未実証で、単一seedの結果から統計的な優越を主張しない。
同じDPT構造・seedでもCNNの初期化乱数消費順が異なるため、DPT初期値の完全一致は保証しない。


診断停止は`DIAGNOSTIC_STOP.json`へ記録し、全epoch終了までは`COMPLETED.json`を作らない。
runnerの`failure.json`はCPUのclip/frame情報とエラー表面化時の処理段階を保存する。
CUDAエラーは非同期に表面化し得るため、段階の記録だけで原因を断定しない。
serial/overlapが混在する比較では各runの方式を結果・図へ表示し、速度差をCNNだけの効果と扱わない。
事後学習probeも本学習と同じ先読み設定を使い、receiptの一致を確認する。
