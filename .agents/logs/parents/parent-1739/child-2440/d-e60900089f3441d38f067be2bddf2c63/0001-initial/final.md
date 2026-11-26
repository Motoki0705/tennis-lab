[#2450をドラフトで作成しました。](https://github.com/Motoki0705/tennis-lab/pull/2450) ベースは #2448、ブランチは `codex/parent-1739-issue-2440-primary`、HEADは `068076542bace61ef15c2e4f92935d5f714ac624` です。CIは実行中です。

- 成功55クリップの20,804視点観測を再集計し、選別落ち10,853件を再現しました。領域内外の内訳は元の1 mm丸めで一致し、丸め前は境界の4件で分類が変わりました。[再現手順と集計](https://github.com/Motoki0705/tennis-lab/blob/068076542bace61ef15c2e4f92935d5f714ac624/knowledge/runs/run-i2440-player-selection/README.md)
- 保存済みの追跡・姿勢・外観特徴を使った全69クリップのCPU再生では、**ダブルス幅・滞在25%**で3D選手フレームが13,218件増えました。012・020・058が三角測量まで進み、新規停止は0件、停止は14件から11件に減りました。
- Meiji 3camの開発用4クリップでは、幅の拡張による採点の変化は0件でした。滞在率を12.5%にすると、選手boxが348件増え、非選手boxも212件増えました。

既存経路ではダブルス幅・25%を追加採点の第一候補とします。#1739の新経路では、短い断片を複数視点と前後の3D軌跡で補えるかを検証します。#2442の全体ラベル公開後に、4カメラの誤選別と人手抜き取り結果を確認します。

ローカルでは108テスト、pre-commit、knowledge検査が通過しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2450,"head":"068076542bace61ef15c2e4f92935d5f714ac624","resume_on_success":false}}