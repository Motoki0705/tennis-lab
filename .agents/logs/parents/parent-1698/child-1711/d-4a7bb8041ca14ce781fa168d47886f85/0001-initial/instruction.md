## この委任の位置づけ

親 #1698 の子。先行の #1706 で `video_ball_annotation.v3` の契約・reader・利用判定が `src/tasks/ball_detection/annotation/video` に入った（PR #1742、merge済み）。この子は、保存済みのv3注釈と、保存済みのコート検出・校正・向き・配置（固定幾何）を、sceneの正式な成果物として取り込み、推測点を含めて三角測量するところまでを接続する。sceneの一連実行への配線は #1713 が行う。GPT workerの実行には依存しない。

## 実装してほしいこと

担当issue本文の変更範囲と受入条件をすべて満たし、PRを作成するところまで行う。mergeは親が行う。

- v3の読込と利用判定は #1706 の reader と usage（位置利用可・直接観測・推定の各判定）をそのまま使う。scene側にv3のschemaや状態の意味を複製しない。
- BallAnnotationImporter は v3 注釈から scene のball成果物を作る。FixedGeometryImporter は固定bundleから court_detection、court_calibration、court_side、camera_alignment を登録する。どちらもGPTの起動、コートの再検出や再校正、学習データの公開を行わない。元bundleと注釈の出自を新しい成果物に記録し、旧geometryのmetadataを上書きしない。
- 位置利用可のmaskと直接観測のmaskを分けて保持し、推定点も三角測量に使う。推定点を observed として偽装しない。既存のモデル推論経路の意味（実観測だけを使う ball_points の契約）は変えない。
- 利用できる座標が2view未満の時刻は欠測とし、寄与したviewと、各viewの点の由来を保存する。旧ball_pointsや旧3D結果を流用しない。
- GPT注釈の点に対する方針: 観測と推定は同じ重みで扱う（親issueでの初期設計案）。補間秒数の上限や低い重みを新たに持ち込まない。平滑化はかけない。再投影誤差などで点を外す処理を入れる場合は、外した点と理由を成果物に記録し、黙って落とさない。選んだ方針はREADMEとPR本文に書く。
- 実データの固定bundleは `outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/fixed_input_bundle_v1/`（`bundle.json` と `scene.json`）にある。読込形式の設計時に構造を読んでよい。ただし、テストは小さな合成fixtureで行い、実データでの受入（3030件の注釈・三角測量）は後続 #1715 の範囲なので行わない。

## テスト

tests/README.md のテスト作成方針に従い、担当issueの各受入条件の「検証方法・証拠」に書いたテストを実装する。既知の3D点と非対称なカメラから独立に作った2D fixtureで、片方のviewを推定点にしたときの期待3D、寄与view、由来を検証する。旧経路の ball_points の契約テストは残す。

## 実行環境の制約（重要）

このマシンはGPUなし、RAM約4GBのラップトップ。`pytest -n auto` を全体で走らせるとメモリが尽きる恐れがある。変更範囲のテストは `-n 0` または `-n 2` で、対象ディレクトリに絞って実行する。全体テストはCIに任せる（PRへのpushごとに実行され、`Python` jobの成功がmerge条件）。PR作成後にCIの結果を確認し、失敗があれば修正する。GPUを使うモデル推論は実行しない。

## 作業量と終了条件

上記すべてとPR作成、CI成功の確認までを1回の作業単位とし、目安は3時間以内。途中で止めてよいのは次のいずれかの場合だけ。その場合は、具体的な論点と選択肢を最終報告に書く。

1. 親issueの合意と矛盾する判断が必要になった。
2. 担当範囲外（GPT worker、自動継続、dataset公開、sceneの全工程CLI）の変更が避けられなくなった。
3. CIの失敗が本変更と無関係で、自力で解消できない。
