# 実装前のテスト設計（#1558）

対象は診断であり、production方式と既定設定は追加・変更しない。固定bundle、runner、
reader、triangulation、採点、比較媒体は #1669 の部品を使う。

| 契約 | 検出する故障 | 層・既存node・追加 | コスト上限 |
|---|---|---|---|
| 同じcamera/frameの共通採点集合で差を測る。欠測・別foldを混ぜない | 配列順のまま引算、coverageの増減を改善と誤認 | unit: `test_scoring.py` の手計算nodeを再利用しpaired集計を1件追加 | 新規1件、1秒以内 |
| signed errorは予測−参照、中心化したnative位置への回帰。25frame単位のbootstrap CIと不足支持を明示 | 符号反転、camera/frame入替、空集合を0と報告 | unit: run scriptの数値診断。非対称な3camera/時刻、既知傾きで1件 | 新規1件、1秒以内 |
| K=1..8回収率はargmax誤差>20pxの点が分母。candidate_validと<=20pxを尊重 | 無効slot=0を正解と扱う、>と>=の逆転、別camera参照 | unit: scriptで具体的な20/21pxと欠測候補、1件 | 新規1件、1秒以内 |
| 注釈速度の差分と時間補間は連続有効点だけ、欠測・split・端を跨がない | 欠測補完、後半ラベルを前半fitへ使用、外挿 | unit: scriptに非連続maskと異なる傾き、1件 | 新規1件、1秒以内 |
| LOESSの前後は同一採用mask。イベント窓は元camera行の和集合、飛行中は補集合 | camera別イベントの多重カウント、推定点の観測化 | unit: scriptに非対称誤差と重複/1frame差イベント、1件。既存 `test_camera_event_rows_use_union_windows_without_multiplying_residuals` を再利用 | 新規1件、1秒以内 |
| offsetはcam0を0に固定、各camera絶対値<1frame。fitは[0,505)だけ。診断oracleはheldout時刻に合わせる | 符号、秒/frame、camera順、後半によるfit変化、heldout時刻を0扱い | unit: 同深度・異なる内部/外部cameraと既知0.3/-0.4frameの解析軌道、fit/診断接続を2件 | 新規2件、各5秒以内 |
| 実baseline出力と入力隔離は基盤契約を保持 | labelが方式入力へ流れる、保存済み3Dへ依存 | 既存 `test_baseline.py`、`test_method_runner_excludes_labels_reference_and_heldout_camera`、`test_ball3d_bundle.py` を実行 | 既存資産不要、各5秒以内 |

新規は当初8ケース以内、pytestは `-n 0`、importを含むwall予算30秒。
`--collect-only` と `--durations=0 --durations-min=0` のnode別記録を残す。
fixtureは既存の軽いcamera/reference factoryを再利用し、重いbundleはCIへ入れない。

実データはCPU・thread=1で逐次実行する。baselineとLOESSなしは共通runner、
annotation oracleは採点専用入口。時刻fit・候補選択oracleはrun内限定で診断と明示し、
方式登録もMethodInputへの注釈変換も提供しない。時刻fitは前半のみ、後半は固定した値で採点。
3Dの絶対的誤差と、校正/注釈の誤差の分離は独立3D正解がなく識別できないため、
誤差予算は非加算的なcounterfactual差とcoverageを併記する。

## mainへの積み直し（#1669レビュー修正後）

paired集計も `FixedInputBundle` を受け取り、基盤の `error_samples` のidentity照合を
迂回しない契約にする。既存の非対称な共通集合テストをbundle fixtureへ移行する。
`test_paired_scoring_rejects_results_from_other_bundle_or_timeline` をunitに2ケース追加し、
別bundle digestと短いframe軸のLOVO結果が拒否されることを検証する（各1秒以内）。
正解のclip/frame/video不一致は基盤の既存3ケースを拡張し、paired入口でも拒否を確認する。
既存8ケースの数値・符号・欠測・heldout隔離契約は維持する。
共通LOVO cohortと表は基盤の既存unit/integrationを再利用し、同じテストを複製しない。

実データは保存済み24結果をchecksum・型・bundleへ照合して再採点し、旧scoresの既存fieldと
pairwise内訳を全区間・全foldで厳密比較する。解析コードとprediction計算部品の差分も確認する。
新しい全6結果共通cohortを併記し、既存動画を再利用したHTML/指標図を新しいv2へ生成する。
予測replay・時刻fit・動画encodeは再実行せず、v1のbytesは変更しない。
pytestは `-n 0`、収集数とstage別時間を再記録し、他workerと重い処理を並列に走らせない。
