# Court/ball bundle adjustment

[English](README.md) · [简体中文](README.zh-CN.md)

標準pipelineで `ball_reconstruction.calibration=court_ball_ba` を指定する。
既定の `fixed` は従来のcomponent・schema・数値出力を使う。任意の精密化はball経路に
適用し、人物は元のalignmentを使う。ball artifactはschema v2で、scene metadataにも
ball専用のcameraと時刻オフセットを保存する。

`bundle_adjustment.py::refine_court_ball` はcameraごとの7変数（焦点距離の共通倍率1、
姿勢3、camera位置3）、相対時刻、対応ごとの3D点を同時最適化する。主点・縦横比・
歪みゼロを固定し、歪み補正済み画素を扱う `PinholeCamera` の契約を維持する。
空間gaugeはメートルCourtKP14、時刻gaugeは選択camera中の辞書順先頭を0とする。
cameraのframe tは世界時刻 t/fps + offset_secondsを観測する。LOVOは保持viewだけで
独立にfitし、除外cameraの校正・時刻は推定しない。foldごとに時刻の基準も変わる。

raw court型と歴史的diagnosticsのreaderは共通の `court.py` にあり、評価側も同じものを
importする。fitには元homographyの採用点かつvalidなKPだけを使い、それ以外のvalidな
raw KPをholdoutに残す。元の最終Hは元templateへ戻っており、PROSACの `x_reflected` を
二重適用しない。決定済みのhalf-turnだけを適用する。非共線4点なら7変数へ8残差以上を
与え、さらに数値Jacobianのrank=7を要求する。raw LINEは要求しない。holdoutも過去の
検出・点選択・初期校正に条件付けられており、人間の独立したcourt正解ではない。

ball対応は標準consensus・LOESS・三角測量のinlierから作る。fitするviewでは前後の
frameもinlierを要求し、時刻を変えて残差を消せないよう支持集合を固定する。clip全体を
等間隔に最大 `max_ball_frames` 件へ制限し、動きと基準cameraへの接続を要求する。
fractional timeは隣接した採用2D観測の区分線形補間で扱う。最終出力はt-offsetで再標本化し、
従来の三角測量・速度gateを実行する。端の外挿と欠測を跨ぐ補間はしない。coverageが下がる
場合があり、fit用3D点で出力の欠測を埋めない。
ball artifactと `SceneResult.ball_uv/ball_vis` は元frameの2D座標・maskを保持する。
再標本化した座標・可視性は3D計算と明示的な `resampled_uv_px/resampled_visibility` 診断にだけ使う。

court/ballのpixel scaleで割った残差へsoft-L1を使う。scoreを較正確率とは扱わない。
priorの標準偏差はlog焦点0.05、姿勢0.02 rad、位置0.5 m、時刻0.5 frameで、共通の
dimensionless prior weightを掛ける。cameraの境界はprior単位の4倍、時刻は
`max_offset_frames < 1`。scale・prior・CPU予算は `ball_reconstruction.bundle_adjustment`
で指定する。最適化失敗・退化・支持不足は例外で停止し、別の結果へ戻らない。

6項目の既定値はHydraのpipeline compositionが所有する。`BundleAdjustmentConfig` と
評価adapterの設定は必須引数で、replayも同じYAMLをcomposeし、
`PipelineRuntimeConfig.from_config` を通して方式へ明示的に渡す。

`CalibrationRefinement` にcamera順、秒単位offset、元fit frame、solver・支持・境界の
診断を保存する。評価adapterは既存の `Ball3DResult` とtyped runnerへそのまま渡す。
GT・正解mask・動画パス・除外cameraのevidenceはsolverの入力に含まれない。
clipの結果・raw KP holdout重ね描き・再現手順は `knowledge/runs/` に記録する。
