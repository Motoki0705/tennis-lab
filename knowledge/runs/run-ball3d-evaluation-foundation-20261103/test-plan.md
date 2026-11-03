# #1556 実装前のテスト設計

親 #1554 の2026-11-03合意と #1556「基盤のテスト設計（実装前）」を適用する。
実装は既定Codexの共通ランナーで行い、推論・GPU・再注釈は行わない。

| 契約 | 検出したい不具合・期待値の根拠 | 層・既存node | コスト上限 |
|---|---|---|---|
| 固定入力のcamera/frame・px・fps・mask・出自を保持し、破損を拒否 | 非対称camera ID/焦点距離/欠測frameを用い、camera入替、frame shift、mask改変、checksum不一致、注釈originの混入を故障注入する。descriptorとsourceの独立記録が期待値 | unitを基本とし、component codecからbundleへの保存読込だけintegration。既存 `tests/unit/tennis_scene/pipeline/test_component_store.py::test_source_mismatch_cannot_reuse_another_clip` と `tests/unit/tennis_scene/pipeline/components/test_ball_evidence.py::test_artifact_roundtrip_is_exact_and_load_only_does_not_infer` を実行 | 新規30 case以内、parametrize 50以下、unit各5秒/setup1秒以内を目標。collect-onlyとdurations全件を記録 |
| 方式へ注釈・reference 3D・除外cameraを渡さない | spy方式で実際のrunner入力を確認し、別annotationで採点しても方式結果が同一。可視性maskはdetector由来のみ | unit。runnerと結果の保存読込の接続は小さなintegration | 同上。動画・checkpoint不要 |
| LOVOで保持viewだけを採点し前後半と欠測を分離 | 非対称3camera、手で指定した3Dと投影座標から独立に5/13/25px誤差を作る。期待値を手計算し、非保持viewと未知点の巨大な座標が集計へ入らないことを確認 | unit。oracleは独立に指定した画素観測から既知3Dへ戻ることを確認 | 各数frame、全1010frame fixtureを量産しない |
| baselineを実storeから再現 | 保存されたdecode/gate・LOESS設定をCPUで再実行し、単一観測・ball_points・3D位置・有効/使用view maskを保存値と照合。元codeを新codeと偽装しない | 実clipはCI fixtureにせず、knowledgeの実行scriptと数値receiptで別途検証 | 実所要時間を別記。unitの5秒基準と混同しない |
| 表示が欠測を跨がずN方式を扱う | 非対称な欠測を持つ軌跡の線分、全view/LOVOの軸、HTMLのlocal参照を確認。実動画は全frame decodeと代表画像の目視 | 純粋な線分・集計はunit、小動画への出力だけintegration | 最小動画、setup1秒を目標。実clipの動画検査は別計測 |

既存テストの期待値や実装の式をコピーして正解を作らない。合成fixtureは契約検査専用で、
実clipの正解ではない。実注釈未到着なら精度・イベント・oracleの実採点を未実施とし、
detector予測をGTへ流用しない。削除・統合する既存nodeはない。
