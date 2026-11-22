## この委任の位置づけ

親 #1698 の必須子7件の最初の1件。後続の #1707（注釈要求登録）・#1708（白紙注釈worker）・#1709（自動継続）・#1711（scene import・三角測量）・#1712（dataset公開）が、この子で決めるv3契約・reader/writer・利用判定を共通の正本として使う。後続が状態の意味を再定義しなくて済むよう、意味と判定APIを明確にすることを最優先にする。

## 実装してほしいこと

担当issue本文の変更範囲・受入条件をすべて満たし、PRを作成するところまで行う（mergeは親が行う）。

- `video_ball_annotation.v3` のschema・writer・reader・検証・共通利用判定を `src/tasks/ball_detection/annotation/` 配下に置く。既存のv2の定義箇所（`src/tennis_scene/pipeline/imports/ball_annotations.py`、`src/tasks/ball_detection/generate_dataset/frame_store/sources/meiji.py`、`src/tasks/ball_detection/dataset_statistics/inputs.py` など）を読み、v2の意味とv3の対応表を作る。今回はv2の読み手を書き換えない（scene接続は#1711、dataset接続は#1712）。
- 状態は少なくとも「直接観測」「推測・補間による座標あり」「遮蔽（座標あり/なし）」「out_of_frame」「no_active_ball」「最終unknown」「未処理」を区別する。遮蔽の有無と推測座標の有無は直交させる。
- 利用判定は別々の関数・maskで返す: 処理済み / 位置利用可 / 直接観測 / 推定 / active_ball教師の負例候補。no_active_ballは「プレー中の対象球がない」というactive_ball専用の意味であり、全ボール検出の負例とは別であることを `target_semantics` で表す。
- 集計API: 未処理・最終unknown・座標利用不可・直接観測・推定などの件数を返す。親issueの後続受入条件「座標利用不可81件以下・最終unknown25件以下」を同じ意味で数えられるようにする（座標利用不可には、座標のない遮蔽・画面外・no_active_ballを含め、未処理は別に数える）。
- 画面外の推定座標（画像外の座標）を保存できるかは設計判断。保存を許すなら「画像外である」ことを判定できるようにし、黙って位置利用可へ混ぜない。どちらを選んだかと理由をREADMEとPR本文に書く。
- 推測点の重み・平滑化は#1711/#1712の範囲。v3には由来（観測/補間/遮蔽推定など）を保存し、重みは持たせなくてよい。持たせる場合は理由を書く。
- v2→v3の対応表（保持できる情報・復元不能な情報）をREADMEに書く。v2のunresolvedをv3のout_of_frame/no_active_ballへ推測変換しない。実データ変換は後続#1714なので行わない。

## テスト

tests/README.md のテスト作成方針に従う。担当issueの各受入条件の「検証方法・証拠」に書いたunitを実装する。camera・frame・PTSを非対称にし、状態表は仕様から独立に手計算した期待値で照合する。

## 実行環境の制約（重要）

このマシンはGPUなし・RAM約4GBのラップトップで、`pytest -n auto` は全体で走らせるとメモリが尽きる恐れがある。変更範囲のテストは `-n 0` か `-n 2` で対象ディレクトリに絞って実行する。全体テストはCI（PRのpushごとに実行、`Python` jobの成功がmerge条件）に任せ、CIの結果をPR作成後に確認して、失敗があれば修正する。

## 作業量と終了条件

上記すべてとPR作成・CI成功の確認までを1回の作業単位とし、目安3時間以内。途中で止めてよいのは、(1) 親issueの合意と矛盾する判断が必要になった、(2) 担当範囲外（scene・dataset・worker）の変更が不可避になった、(3) CIの失敗が本変更と無関係で自力で解消できない、のいずれかの場合だけ。その場合は具体的な論点と選択肢を最終報告に書く。
