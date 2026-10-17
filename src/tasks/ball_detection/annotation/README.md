# Ball 注釈と学習循環

共通の[注釈基盤](../../base/annotation/README.md)に、ball固有の予測・レビュー・
教師公開を接続する。新規clipへの座標・hybrid推論は以下、差分編集・画像確認・教師採用の仕様は
[review](review/README.md)を正本とする。

`BallPredictionAdapter` を共通 `Predictions(layout, plan).run(manifest, adapter)` に
渡す。収集台帳に準備済みとして登録されたclipと、そのcampaignに固定したモデル・
推論設定だけを受け付ける。既存予測はhashとframe対応を検証して返し、モデルを
ロードしない。dataset公開済みclipの通常再推論や教師上書きは行わない。

- `MDDCheckpointSpec` はcheckpoint schema、全architecture、MDD入力契約、学習manifest
  digestを固定する。既存の `restore_coordinate_model` で全重みをstrict復元し、
  MDD-only query modelであることを確認する。pose依存modelを別方式に読み替えない。
- `MDDInferenceConfig` / `HybridInferenceConfig` は32frame、stride、重複処理、復号方式、座標系、device、精度を
  固定する。JPEG準備では `opencv_jpeg_rgb24_native`、既存動画入力では `pyav_rgb24_native` を指定し、
  manifestの保存形式と異なる設定を拒否する。保存済みJPEGをOpenCVでnative解像度のRGB uint8へ復号し、
  推論時のresize/JPEG再圧縮を行わない。媒体と削除・再取得の契約は
  [collection](../../base/annotation/collection/README.md#jpegを残して動画を削除する保存方式)を参照する。
  窓の先頭を基準とする実source PTSの秒をfloat32で渡す。MDD前処理の正本は既存model。
- 最後の窓は末尾の実32frameから始め、全frameを覆う。32frame未満は理由付きで
  拒否し、padding/同一frameの複製は行わない。通常の15秒選別clipで不足することを
  想定した処理ではなく、壊れた・例外的な入力の検証である。
- 同じframeの重なりは窓中心までの距離が小さい出力、同距離なら早い窓を採る。
  source/frame/PTS/window開始位置を全行に保存する。座標変換は
  `x=u*(width-1), y=v*(height-1)`。対象外の前後文脈frameも予測行を持つが、教師の
  対象範囲はmanifestの `target_range` で別に管理する。
- 座標modelの出力は `coordinates_only_unreviewed`。confidence・球の有無・可視性を作らない。
  対象球や遮蔽等の意味は後続レビューの責務であり、座標だけで教師採用しない。
- 1 adapter instanceに1モデル・1推論設定を保持し、複数clipで再利用する。CPUに
  置くRGBは1窓分と逐次decode中の1frameに限定する。媒体を先頭から順に読み、
  重なる実frameを同じbufferで保持する。末尾窓を含め、窓ごとのRGBとPTSは維持する。
  複数レビューworkerにモデルを
  複製せず、GPU実行は共有training queueと資源admission側で管理する。

保存先は `predictions/ball/<clip-id>/<request-digest>/` の `coordinates.json` と
共通 `receipt.json`。座標保存後・receipt保存後の中断では、既存artifactを検証して
同じ結果を台帳へ回収する。別モデル/設定/manifestの既存登録を変更して再実行しない。
検証用データへのモデル評価は通常の注釈推論と別の入口を使う。

単体検証は実VFR MP4の窓・PTS・native座標変換と、小さい実MDDのcheckpoint復元・
CPU forwardを含む。採用候補の実モデルと実テニス映像の定性確認はpilotで行う。

## Hybridの予測案

3段目の `mdd_hybrid_training.v1` は `HybridCheckpointSpec` と
`HybridInferenceConfig` で明示的に選ぶ。model specはarchitecture・5ch入力契約・
学習manifest digest、推論設定は `presence_threshold` と `subpixel` を必須とする。
存在閾値は採用checkpointのrecipeと一致させる。全重みをstrict復元し、親checkpointや
学習datasetを開かない。座標modelとhybridをschemaで識別し、別familyへフォールバックしない。
両familyは同じ台帳・窓・decode・再開経路を使う。保存済みの座標model campaignも固定版を保つ。

保存先は同じrequestディレクトリの `hybrid.json`。`HybridPredictions` は
`heatmap_presence_unreviewed` とnative 1/4格子の `(height,width)` を持つ。
各行の `presence=present/absent`、生の `presence_logit`、FP32 sigmoidの
`presence_score`、存在gateを掛けない `heatmap_peak` を保存する。
両scoreは未較正で、画像確認済みや教師採用の判定ではない。present時だけheatmap由来の
native `x,y` を保存し、absent時は両方nullにする。NaNをJSONへ書かない。
heatmapのdense全格子は予測案に複製せず、格子寸法とpeak値を保持する。
復元時・再利用時にfamily、格子、frame/PTS、座標範囲、存在score/logit/閾値の整合性を検証する。

preview・レビュー画像では不在予測に点を描かない。workerへの予測状態と確認方法は
[review](review/README.md)を参照。既存のレビュー方針をモデル変更だけで緩めず、
新モデルの代表clipをpreviewしてから、新しいcampaignにモデル・設定・方針の参照を固定する。
進行中campaignのモデル交換や既存clipの再推論には使わない。

`test_hybrid_annotation.py` は合成重みの小型5chモデルをCPU復元し、VFR動画/JPEGから
予測案、画像提示、修正、負例教師の証拠検証まで確認する。学習済み重みと実テニスclipでの
GPTレビュー品質は別の受入確認であり、この合成試験の結果で代用しない。
