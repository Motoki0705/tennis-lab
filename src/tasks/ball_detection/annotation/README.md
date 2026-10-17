# Ball 注釈と学習循環

共通の[注釈基盤](../../base/annotation/README.md)に、ball固有の予測・レビュー・
教師公開を接続する。ここでは新規clipへのMDD座標推論を定義する。

`BallPredictionAdapter` を共通 `Predictions(layout, plan).run(manifest, adapter)` に
渡す。収集台帳に準備済みとして登録されたclipと、そのcampaignに固定したモデル・
推論設定だけを受け付ける。既存予測はhashとframe対応を検証して返し、モデルを
ロードしない。dataset公開済みclipの通常再推論や教師上書きは行わない。

- `MDDCheckpointSpec` はcheckpoint schema、全architecture、MDD入力契約、学習manifest
  digestを固定する。既存の `restore_coordinate_model` で全重みをstrict復元し、
  MDD-only query modelであることを確認する。pose依存modelを別方式に読み替えない。
- `MDDInferenceConfig` は32frame、stride、重複処理、復号方式、座標系、device、精度を
  固定する。RGBはPyAVでnative解像度のuint8へ復号し、resize/JPEG再圧縮を行わない。
  窓の先頭を基準とする実source PTSの秒をfloat32で渡す。MDD前処理の正本は既存model。
- 最後の窓は末尾の実32frameから始め、全frameを覆う。32frame未満は理由付きで
  拒否し、padding/同一frameの複製は行わない。通常の15秒選別clipで不足することを
  想定した処理ではなく、壊れた・例外的な入力の検証である。
- 同じframeの重なりは窓中心までの距離が小さい出力、同距離なら早い窓を採る。
  source/frame/PTS/window開始位置を全行に保存する。座標変換は
  `x=u*(width-1), y=v*(height-1)`。対象外の前後文脈frameも予測行を持つが、教師の
  対象範囲はmanifestの `target_range` で別に管理する。
- 出力は `coordinates_only_unreviewed`。confidence・球の有無・可視性を作らない。
  対象球や遮蔽等の意味は後続レビューの責務であり、座標だけで教師採用しない。
- 1 adapter instanceに1モデル・1推論設定を保持し、複数clipで再利用する。CPUに
  置くRGBは1窓分と逐次decode中の1frameに限定する。複数レビューworkerにモデルを
  複製せず、GPU実行は共有training queueと資源admission側で管理する。

保存先は `predictions/ball/<clip-id>/<request-digest>/` の `coordinates.json` と
共通 `receipt.json`。座標保存後・receipt保存後の中断では、既存artifactを検証して
同じ結果を台帳へ回収する。別モデル/設定/manifestの既存登録を変更して再実行しない。
検証用データへのモデル評価は通常の注釈推論と別の入口を使う。

単体検証は実VFR MP4の窓・PTS・native座標変換と、小さい実MDDのcheckpoint復元・
CPU forwardを含む。採用候補の実モデルと実テニス映像の定性確認はpilotで行う。
