# #1565 実装前の設計（2026-11-03）

親 #1554 revision 6、子 #1565 の本文・全コメント、AGENTS、memory、対象README、tests/README、
基盤PR #1669 と先行PR #1670/#1672/#1673・knowledgeを確認した。

## 契約・不具合・層・既存node・コスト

| 守る契約と期待値の根拠 | 検出したい不具合 | 最軽量の層 | 既存nodeと追加の理由 | コスト上限 |
|---|---|---|---|---|
| 重力はcourt Z下向き、抵抗は -k\|v\|v。独立DOP853で生成した非対称多視点の既知drag/offset/bounceを復元 | 力の符号、秒/frame、camera軸、drag脱落、イベントを平滑化する変更 | unit/tennis_scene/ball_trajectory | 新規。baseline testは物理/時刻を扱わない | 2形式・合計4数値case、各5秒目安、超過は実測と理由を記録 |
| GTなしのbaselineから区間/イベントを検出し、欠測出力を保持する | 長いgap横断、invalid座標のfit混入、bounce/hitの見落とし | 同unit | 新規。既存ball_pointsの欠測契約を再利用 | 3case以内、各0.5秒 |
| 設定・軸・証拠の欠落を明示拒否、収束状態と識別不能な不確かさを明記 | 静かなbaseline fallback、camera入替、証拠なし実行 | 同unit | 新規。基盤test_contracts/test_courtはそのまま利用 | 不正入力8case以内、各0.1秒 |
| 既定のdefinition→assembler→component出力がmain由来の固定配列と厳密一致 | opt-in設定による既定の数値/観測mask変更 | integration/tennis_scene/test_ball3d_segment_physics.py | 既存test_ball_points::test_filled_and_rejected_points_cannot_supply_the_second_real_viewとtest_orchestrator::test_runtime_dependencies_come_from_component_declarationsも実行 | 1case、5秒以内。期待値は変更前mainから生成 |
| 選択方式が実pipelineと基盤adapterで同じ結果、LOVOは選択2viewのみ、offset/共分散/イベントを保存復元 | 設定が届かない、heldout参照、診断の保存漏れ | 同integration | 基盤test_scoring::test_method_runner_excludes_labels_reference_and_heldout_camera、test_ball3d_bundle::test_real_component_codec_bundle_and_cpu_runner_roundtrip再利用。新規は方式固有接続 | 1case、5秒目安 |

CPU・OMP/MKL 1thread、pytest -n 0（最大でも2）。fixtureはテスト内の小さい数値配列、setup1秒以下。
新規は最大18caseを想定。--collect-onlyと--durations=0 --durations-min=0の実測を保存する。
物理の符号またはoffset符号を一時反転し、独立期待値のassert失敗と復元後成功を確認する。

## 実データの計画と設定固定

固定bundleだけをloaderで読み、既存runnerで全view・各LOVOを実行する。候補点MAP残差と
dense map尤度を比較する。baselineは初期値と区間検出にだけ使用し、GTは方式へ渡さない。
cameraの辞書順先頭をoffset=0に固定し、残りを絶対値0.99frame以内とする（LOVOでも保持cameraだけ）。
gravity=9.81、clip共通dragを推定。短gapの内側をfitしても、出力validはbaselineの実観測支持を超えない。
候補・dense両形式の設定は合成検証後・GT採点前にcommitして固定する。ラベルを使う調整は行わない。
後半[505,1010)の主指標、前半と全区間、LOVO別、採点数/coverage・共通cohortを併記する。
比較scriptと実行証拠はこのrunへ、予測・動画・HTMLは新規outputs directoryへ保存する。

## 実測によるコスト設計の更新

新規は13case（unit11、integration2）。接続integrationは初回計測で28.99秒だった。
76frame・2飛行の小さな入力で、実pipelineのfit、全view/3つのLOVO、除外view改変後の再fitの
6回を実solverで通し、実codecの保存復元も検証する。mockに置換すると同じsolverの使用、
view別の時刻基準、除外cameraへの非依存という固有の数値契約を失うため、この1caseの
5秒超過を残す。setupは0.006秒。収集99件・実行99passed（pytest表示58.56秒）を確認後、
不確かさのrank検査を列正規化して単位への依存を除く修正を行うため、同じ一式で最終計測する。

## CI修正の実装前設計（PR #1679）

1. 守る契約: 設定の既定値はHydra compositionだけが持ち、Python adapterは必須引数を受ける。
   同一の初期3Dと観測をcamera軸だけ並べ替えても、solver入力はcamera ID順で一致し、
   位置・offset・共分散・使用viewと診断は元のcamera軸へ正しく戻る。
2. 検出したい不具合: 新規configのPython既定値の再導入、設定の暗黙生成、
   残差行/parameter列の順序によるLSMRと終了判定の変動、offset共分散/診断の軸復元漏れ。
3. 層: 新しい重いテストは追加せず既存unitの順序検証を厳密一致・非対称mask・共分散へ拡張。
   既定値/overrideの接続は既存configuration integrationと本方式integrationを再利用する。
4. 既存node: `tests/unit/utils/configuration/` 全件（特に `test_audit.py::test_repository_source_satisfies_current_audit`、
   `test_contracts.py::test_catalog_has_exact_source_declaration_parity`）。
   `test_joint_fit_recovers_signed_offsets_drag_bounce_and_camera_reordering`、既定のmain配列一致、
   opt-in component/codec/LOVOの既存nodeも実行する。テストfixtureはcompositionから明示設定を作る。
5. コスト: pytest -n 0、OMP/MKL/OpenBLAS各1thread。追加caseは最小のcomposition契約1件まで。
   既存の31秒integration超過理由は維持し、全nodeをcollect-onlyと全phase時間で再計測する。

修正前に2つの順序で初期値・bounds・残差/Jの並び・終了理由を記録する。
許容誤差とsolver終了閾値は広げず、camera ID順への正規化で原因を除く。
実データでは全viewと全LOVOのcamera IDが既に辞書順であること、composition値がv3と一致することを
確かめ、CPUで全方式/全foldを再実行し、予測配列と採点をv3へ照合する。
変化があればv3を上書きせずv4へ、変化がなければ差分とchecksumの検証receiptを保存する。
