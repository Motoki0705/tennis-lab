## この委任の位置づけ

親 #1698 の最後の必須子。先行の6件はすべてmerge済みで、個別のAPIがそろっている。

- #1706（PR #1742）: v3の契約・reader・利用判定・集計（`src/tasks/ball_detection/annotation/video`）。
- #1707（PR #1745）: ローカル動画の白紙注釈要求の登録 `configure_local_videos()`。
- #1708（PR #1763）: 担当cameraの画像だけで注釈するGPT worker、途中保存、完成の確定、探索方針の版と比較計測。
- #1709（PR #1766、merge済み）: 90分の時間切れからの自動継続。`src/tasks/annotation/local_controller.py` の `LocalAnnotationController(layout, runtime)` が登録済みの全cameraを実行し、`cycle()` の戻り値の `reviews` でcameraごとの `state` と `completion`（`VideoCompletion`、その `annotation` が保存済みv3注釈）を返す。呼出元のschedulerが終了までcycleを進める。dataset公開はこのcontrollerの完了条件に含まれず、planの `publish_dataset` を後続へ渡す設計になっている。明示復旧は `recovery.recover(controller, RecoveryInput(...))`。詳細は `src/tasks/annotation/README.md` の「白紙注釈の自動継続」を読む。試験用の論理時計 `test_clock` もある。
- #1711（PR #1758）: BallAnnotationImporter・FixedGeometryImporter と、推定点を含む三角測量。
- #1712（PR #1761）: active_ball のdataset snapshotの公開と学習reader。

この子は、これらをsceneの公開入口へつなぎ、1回のworkflowで「登録 → 注釈 → 自動継続 → 保存と再読込 → 固定幾何での三角測量 → 選択した場合だけdataset公開」まで進むようにする。そのうえで、今回の完了範囲（実装と小規模テスト）を証拠付きで閉じる。

## 実装してほしいこと

担当issue本文の変更範囲と受入条件をすべて満たし、PRを作成するところまで行う。mergeは親が行う。

- ball_detection の方式選択（既定は通常のモデル推論、明示オプションでGPT白紙注釈）、固定幾何bundleの参照、dataset公開オプションを定義する。既定の通常推論経路が変わらないことを回帰確認する。
- 注釈の保存、sceneの完了、dataset公開を別々のstatus・receiptとして記録する。公開だけが失敗した場合は、注釈をやり直さずに公開だけを再開できるようにする。公開なしは成功として扱う。追加学習は起動しない。
- 探索方針の版と計測出力（#1708）を公開入口につなぐ。比較用のrunは通常のdataset公開と区別し、自動で公開しない。
- 後続の実動画受入（#1715）で使う集計を実装する: 3cameraの全件の合計で、座標利用不可・最終unknown・未処理の件数と、camera別・状態別の内訳を出す。判定条件は「座標利用不可81件以下、かつ最終unknown25件以下、かつ未処理0件」。座標利用不可には、座標のない遮蔽・画面外・no_active_ballも含める。参照データの `source_status=unknown`（補完後に推定座標として採用された点を含む）を最終unknownとして数えない。小規模fixtureでは、この本番の件数を合格条件に流用しない。
- 利用手順、復旧手順（公開だけの再開を含む）、後続の実データ移行（#1714）と実動画受入（#1715）に必要な入力をREADMEにまとめる。実動画の固定bundleは `outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/fixed_input_bundle_v1/` にある。#1711 によると、このbundleではcourt_detectionの配列が省略されているので、明示指定 `court_detection="verified_calibration_frame0"` が必要になる。

## テスト

tests/README.md のテスト作成方針に従い、担当issueの各受入条件の「検証方法・証拠」に書いたテストを実装する。代表的なE2Eは1つにする。CPUの短い3camera合成動画と固定幾何を使い、GPTの応答と時間の境界だけをfakeにして、実際のサービス・保存・三角測量・publisherを通す。どこをfakeにしたかを最終報告とPR本文に書く。先行の子が網羅した異常系は繰り返さず、接続の代表例だけにする。

## 実行環境の制約（重要）

このマシンはGPUなし、RAM約4GBのラップトップ。`pytest -n auto` を全体で走らせるとメモリが尽きる恐れがある。変更範囲のテストは `-n 0` または `-n 2` で、対象ディレクトリに絞って実行する。全体テストはCIに任せる（PRへのpushごとに実行され、`Python` jobの成功がmerge条件）。PR作成後にCIの結果を確認し、失敗があれば修正する。実GPTのAPIとGPUは使わない。ローカルでは実PSS計測などの既存テスト数件が変更前のmainでも失敗する。退行と判断する前に、変更前のmainで比べる。

## 作業量と終了条件

上記すべてとPR作成、CI成功の確認までを1回の作業単位とし、目安は4時間以内。途中で止めてよいのは次のいずれかの場合だけ。その場合は、具体的な論点と選択肢を最終報告に書く。

1. 親issueの合意と矛盾する判断が必要になった。
2. 先行の子の契約を変える必要が出て、その変更が担当範囲を大きく超える。
3. CIの失敗が本変更と無関係で、自力で解消できない。
