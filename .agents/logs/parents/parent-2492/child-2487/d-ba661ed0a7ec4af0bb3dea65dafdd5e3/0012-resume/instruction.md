# #2487 統合担当: PR #2499 の正式レビューの major 3件を直す

PR #2499 は子 #2575 を閉じる PR に変わった（親が本文の先頭を `Closes #2575。Refs #2487, #2492, #2495, #2496, #2491。` に変えた）。#2487 は、下流の経路での品質を確かめるまで開けておく。本文を直すときは、この先頭行を保つ。

正式レビュー（run `parent-2492-338fb7c8b563469e8b93be68024cbfc6`、固定 head `f5f1a5f2b`）の結果は major。比較表44行・Pareto 図2枚・速度18条件の P50/P95・保持済み queue ログ50件の SHA-256・標準と FastHMR のキャッシュ識別値は、レビューで一致が確認された。親は次の3件をすべて採用する。

## 1. 同一プロセスで ECPose-X の後に ECPose-L をロードできない
- レビューが実行して確かめた。`load_network` を同じ CPU プロセスで L → X → L と呼ぶと、最後の L が `load_state_dict(strict=True)` で失敗する。上流 EdgeCrafter（固定版 `d8752b75`）の `ecpose/engine/core/yaml_utils.py` の YAML ローダーが前回の設定を共有辞書に残し、L が X の `ffn_ratio=6` を引き継いで、MLP 幅 2304 のモデルを作るため（公式重みは 1536）。
- 直し方: ロードごとに YAML の作業状態を分離する（上流のコードは書き換えない。repo 側のローダーで状態を隔離する）。黙って作り直すのではなく、構築したモデルの設定が選んだ release の設定と一致することを確かめる。
- 回帰テスト: 同じプロセスで L → X → L（とその逆）を切り替えてロードできること。公式重みがないと動かないなら、重みなしで設定の分離だけを確かめるテストと、重みを使う実行確認の両方を用意し、実行確認の結果を PR 本文に書く。

## 2. データセット公開時に ECPose の成果と宣言した backend を照合しない
- 共通の `publish_annotation`（`src/tennis_scene/pipeline/dataset_publication.py`）に、PT32 で作ったシーンと PT16 の publication identity を渡すと、ECPose の実行設定を照合せずに PT16 として公開できる。`completed_annotation` も同じ検査を使い、マーカーが一致すれば実行計画を作らずに再利用する。`component_origin_only` は実行計画でしか効かない。
- 直し方: 公開と再利用の両方で、シーンが参照する BBox と pose が同じ ECPose component に由来し、そのモデル・backend・前処理・decode 設定が公開の宣言に一致することを検査する。#2501 が FastHMR に入れた検査（`body_artifacts.py`・`dataset_publication.py` で `origin == 'component'` と設定の一致を要求）と同じ方針にそろえる。
- 回帰テスト: 別の精度・別のモデルサイズで作った ECPose シーンを公開しようとすると明示的なエラーになること。`completed_annotation` が不一致のマーカーを再利用しないこと。標準 recipe の公開・再利用が変わらないこと。

## 3. RF-DETR-L TRT16 の測定済み PT32 差を共通基準で表示する
- RF-DETR-L TRT16 は、以前の精度測定と今回の速度測定で同じ TensorRT plan（SHA-256 一致）を使っている。保存済みの PT32 比較では IoU P05 が 0.9620 で、#2492 の共通基準（検出 0.98／0.01）を下回る。精度ノードは旧基準で合格と書き、比較表 `conditions.csv` は `PT32未計測` だけを出し、Pareto 図にも出力差が示されていない。
- 直し方（ユーザー判断: 出力差は共通基準で併記し、検出は精度で図に載せる）:
  - 旧判定の記録と、今回の速度測定の `PT32未計測` は消さずに残す。
  - 保存値から求めた共通基準での判定（IoU P05 0.9620 < 0.98 で不成立）を、比較表の列と Pareto 図の注記に加える。疑似GT 精度と速度による Pareto 点は残す。
  - 精度ノードの合格記述に、旧基準による判定であることと、共通基準では不成立であることを書き足す。
  - 同じように、PT32 との比較を保存している他の条件がないか確かめ、あれば同じ扱いにする。
- 表・図はスクリプトから作り直し、数値が保存済みの結果と一致することを確かめる。

## 終わり方
- 関連テスト、境界検査（`tests/e2e/development/test_architecture_boundaries.py`、ローカル）、pre-commit、knowledge 検査を通して push する。PR 本文の検証の節を新しい HEAD に合わせる。
- 最終回答に、3件それぞれの修正 commit・回帰テスト（名前と結果）・作り直した表と図の差分の要点、push した HEAD を書き、新しい HEAD の CI を待つ `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"<新HEAD>","resume_on_success":false}}` を返す。
