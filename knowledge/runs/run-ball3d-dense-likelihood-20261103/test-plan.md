# 実装前の設計（#1564）

親 #1554 の全45コメント・合意revision 6、子本文、#1669、先行 #1670/#1672/#1673 と各knowledgeを確認した。
後半での再調整は行わない。正解は方式入力に含めず、全view/LOVOは基盤runnerだけで実行する。

| 守る契約 | 検出する不具合 | 層・既存node | コスト上限 |
|---|---|---|---|
| 既知の非整数投影から連続3Dを復元 | XY/view/frame交換、格子吸着、logit/log混同、勾配の単位 | 新unit `ball_likelihood/test_likelihood.py`、独立pinhole式・非対称3camera・二次対数山、双線形による量子化との比較 | 3補間×2初期化、各5秒以内、小mapのみ |
| 欠測・画面外・1cameraのdistractorが拒否権を持たない。最低2view支持 | 一様背景なし、無観測を有効化、低信号で架空の3D | 同unit、0/1飽和・全欠測、2camera支持と3番目の誤峰 | 追加5件程度、各1秒以内 |
| 3D加速度priorで飛行中の揺れを抑え、急変と欠測を跨がない | prior未接続、秒/frame混同、GTイベント参照、gap補完 | 同unit、既知の方向転換を含む短軌道、gapと実測のみの境界 | 2件、各5秒以内、9frame窓 |
| 既定設定の出力はmainの固定配列と厳密一致 | 既定方式変更・旧ball_points binding変更 | 新integration `test_ball3d_dense_likelihood.py`、変更前mainの固定期待値を保存。既存 `test_baseline.py::test_without_loess_keeps_acceptance_but_uses_original_detector_coordinates` も再利用 | 1件、12frame、5秒以内 |
| 選択設定→definition→assembler→production→実codecと評価adapterの同一性、LOVO隔離 | 設定無視、順序交換、gate後だけの証拠、除外view漏れ | 同integration。既存 `test_scoring.py::test_method_runner_excludes_labels_reference_and_heldout_camera`、bundle/media integrationを再利用 | 1〜2件、4frame、各5秒以内 |
| 不正設定・camera/frame/サイズ不一致の明示拒否 | 静かなfallback、bool/負の温度・ε、証拠欠落 | 新unit、configuration既存nodeを再利用 | 10同値クラス以内 |

既存の採点・bundle・mediaテストを再利用し、独自loader/scorerや重いE2Eは追加しない。
`pytest -n 0 --collect-only` と `--durations=0 --durations-min=0` を保存する。
故障注入はview順の交換または加速度priorの無効化で数値assertが失敗し、復元後成功することを確認する。

## 資源と探索予算

3×1010×72×128 float32は約106.5MiB。検証済みbundleのmmapを共有し、全体をlogitへ複製しない。
1frameの3view logitは216KiB、9frame窓でも約1.9MiB（splineとoptimizerの作業領域は別）。
コート粗格子は既定で約4千点、細分化は上位16点の周囲だけを2段走査、連続最適化は上位2初期値、最大40iteration。
候補初期化は最大3pair×8×8=192点のbatched DLT。最終目的関数は候補への距離ではなくdense尤度。
各条件を逐次実行し、最初の短いCPU probeで時間/RSSを測定してから全1010frameへ進む。
動画は基盤の逐次reader/writer、pytestは-n 0、BLAS/OpenMPは1thread。
GPU・検出器再推論は使用しない。

## 比較条件（採点前の計画）

bicubic-logit、局所二次logit、局所二次logを同じ候補初期化で比較する。
初期値なしのコートcoarse-to-fineと、3D加速度prior版を別結果として残す。
view別温度と背景混合εの選択は前半の共通採点APIだけを使う。候補数・空間範囲などは事前固定。
前半だけの選択記録を確定してから、後半・全区間を採点する。
誤差・coverage・共通cohortを同時に報告し、先行top-Kのnull状態による大幅欠測を避けるため時間priorは実測支持のある点だけを動かす。

## 最終確認で補う初期値の独立性

coarseの既知位置復元だけでは、将来候補seedへ戻す誤変更を検出できない。
既存の同じ合成fixtureで候補payloadだけを全無効にしたunitを1件追加し、
coarseは既知位置を復元し、候補版は欠測を明示する契約を確認する。
新しいmap fixtureや重いintegrationは増やさず、追加1case・5秒以内を上限とする。

## CI修正の設計

- 設定の正本はcompositionのYAMLのみ。`DenseLikelihoodConfig` の全fieldを必須にし、
  adapterも明示的なconfigを受ける。既存の設定監査directory全体を再利用し、別の免除や監査の弱化は追加しない。
  既存のpipeline接続integrationへ温度/背景率overrideの伝達確認を足す（case数は増やさない）。
- main回帰のpositions/valid/inliers/uv/visibility/reasonsは引き続き完全一致。
  reprojectionだけの差について、float64のfit解から残差を作ってfloat32へ保存する経路と、
  NumPy/BLASのCPU実装を変えたprobeを記録し、float32 ULP単位の上限を決める。
  mainの計算ファイルのGit blob一致も別の証拠にする。画素単位の誤差を隠す許容値は使わない。
- 数値実装とYAML値が不変であること、全保存条件の明示configが一致することを確認し、
  選定denseの1010frame・全viewをCPU再実行してv2の配列と比べる。GT再調整はしない。
- pytestは-n 0で逐次実行。新しい大きなfixtureや重いE2Eは増やさず、collect-onlyと全phase時間を保存する。
- DINO分離の失敗が観測された場合のみ、指定commitの該当1ファイルを同一内容で取り込む。
