# #2601 Clip Studio のクリップ別カウント入力 — 実装依頼

親issue #2601 の合意済み設計を、1本のPR（`Closes #2601`）として実装する。PR本文は日本語。
まず `src/tennis_scene/clip_studio/README.md` と `project.py`・`web/service.py`・`web/app.py`・`web/static/*.js`・`web/review.py` を読み、既存の編集トランザクション（revision検証・atomic自動保存・Undo/Redo 100件）に載せる。

## 合意した設計（変更する場合は理由を最終報告で返す）

### 保存: `projects.json` v2
- `PROJECTS_SCHEMA_VERSION` を2へ。v1は読み込み可能（全clipの `score=null`、`match=null`）。保存は常にv2。v1/v2以外は従来どおりエラー。厳密なキー検査（`_require_exact_keys`）の方針は維持し、v2のキー集合を明示する。
- 動画（video project）ごとに `match`（null可）: `sides`（A/Bの表示名）、`rules`（採点方式）、`first_server`（`A`|`B`）。
- 各clipに `score`（null=未入力）:
  - `status`: `auto` | `confirmed` | `unknown` | `not_point` | `multiple_points`
  - `before`: `{sets:[a,b]（硬式のみ、ソフトテニスでは持たない）, games:[a,b], points:[a,b]}` を整数で保存。表示（0/15/30/40/AD、ソフトの数字、デュース/アドバンテージ）はルールから変換。デュース後の点は正規化した整数で持つ（例 3-3/4-3）。`not_point` は `before` を持たない。
  - `before_source`: `chain`（直前クリップから計算）| `anchor`（手入力）
  - `server`: `A`|`B`、`server_source`: `auto` | `manual`
  - `winner`: `A`|`B`|null（`confirmed` は必須、それ以外はnull）
  - `note`: 文字列。`unknown`・`not_point`・`multiple_points` では非空必須。
- カウントはclipの中に持つので、名前変更・並べ替え・削除で対応が崩れない。新規作成clipは `score=null`。
- 状態とフィールドの組合せ、ルール違反の値（ゲーム終了済みのポイント、マッチ終了後のカウントなど）は保存時（validate）にエラー。静かな補正をしない。

### 採点ルール（共通モジュール）
- #2545（試合統計DB）からも使えるよう、Clip Studio非依存の純粋なモジュールとして切り出す（置き場所は `src/utils` 配下などAGENTS.mdのモジュラー方針で判断）。カウント状態＋得点側→次の状態、サーブ側の計算、表示文字列、検証を提供する。
- ソフトテニス: 4点先取・3-3でデュース・2点差でゲーム。Nゲームマッチ（N=奇数、動画ごとに設定）。ゲーム数が並んだ最終ゲーム（例: 7ゲームマッチの3-3）はファイナルゲーム（7点先取・6-6でデュース・2点差）。サーブ側はゲームごとに交代し、ファイナルゲームでは2点ごとに交代。セットなし。
- 硬式: 0/15/30/40、6ゲームのセット（2ゲーム差、6-6でタイブレーク7点・2点差）。設定: セット数（best of 3/5）、ノーアド有無、最終セット形式（通常タイブレーク[local path]）。サーブ側はゲームごとに交代、タイブレーク中は最初の1点後、以後2点ごとに交代。
- ルールの設定値は明示的なスキーマで保存し、未知の値・欠落はエラー。

### 連鎖（自動入力と一括再計算）
- 順序はclipの開始時刻順。
- 得点側を入力すると `confirmed`（開始前カウントも確認済みとみなす）。終了後カウントを計算して表示。
- 次のクリップが未入力（null）なら、開始前カウントと自動サーブ側を `auto`[local path]`chain` で自動入力。
- `unknown`・`multiple_points` は連鎖を止める（次のクリップは `anchor` 手入力が必要）。`not_point` はカウントを持たず、連鎖はその先へそのまま引き継ぐ。
- 途中を修正したとき（得点側・開始前カウント・サーブ側・状態）、後続の `before_source=chain` の開始前カウントと `server_source=auto` のサーブ側を、確定済みも含めて、次の `anchor` または連鎖停止クリップの手前まで再計算する。各クリップの `winner`・`status`・`note` は変えない。再計算で変わるクリップ数を適用前にUIで表示し確認してから適用する（APIはプレビューと適用を分けるか、変化件数を返す設計にする）。
- `anchor` の開始前が直前クリップの終了後と食い違うときはエラーにせず「不連続」と表示（クリップ化していないポイントがあり得るため）。
- サーブ側の自動計算は `match.first_server` とルールから行い、`manual` は個別に上書き。
- 再計算後にマッチ終了後のカウントになる等ルール違反が出る場合は、どのクリップで何が起きたかを示して適用を拒否する（静かに打ち切らない）。
- カウント編集・試合設定変更・連鎖再計算・CSV取り込みは既存のUndo/Redoとrevisionの1操作として扱う。

### 画面（`web/static`）
- クリップ一覧の各行に、開始前カウント・得点側・状態・不連続の印を表示し、未入力が一目で分かるようにする。
- 選択中クリップの既存「名前・開始・終了」編集欄の下にカウント欄（開始前カウント、サーブ側（自動/手動）、得点側、状態、理由、終了後カウント表示）。消去（未入力に戻す）もできる。
- 得点側はキーでも入力できる（既存の I/O/C/F/Space/矢印[local path],[local path] と衝突しないキー。入力欄編集中は抑制する既存方針に従う）。README のキー一覧に追記。
- 動画ごとの試合設定（A/B表示名、採点方式と設定、最初のサーブ側）を編集できる欄。
- 読取専用レビュー（`__main__.py` の read-only CLI）でもカウントを表示し、編集APIは拒否。

### CSV 取り込み・書き出し
- 列: `clip_name,status,sets_a,sets_b,games_a,games_b,points_a,points_b,server,winner,note`（ヘッダー必須・列は厳密一致）。ソフトテニスでは sets 列は空。
- 取り込みは全件を検証してから一括適用。不正が1件でもあれば何も変えない。存在しないclip名・重複行はエラー。CSVに無いclipは変更しない。取り込んだ行の `before_source`[local path]`server_source` の扱い（anchor/manual 扱い等）を明確に決めてREADMEに書く。1回のUndo単位。
- 同じ形式で書き出し（UIからダウンロード）。往復で値が一致すること。

### 範囲外
dataset `clip.json` への書き出し、音声・3Dからの自動採点、統計DB統合、動画へのカウント焼き込み。

## 受入条件と証拠
1. 保存済みクリップのカウントを入力・変更・消去でき、現在の値を確認できる。
2. 再読み込み後も各カウントが入力先のクリップに対応し（名前変更・並べ替え後も）、他のクリップのカウントが変わらない。
3. カウントのない既存v1プロジェクトを開け、未入力クリップを含めて従来の切り出し操作（作成・修正・削除・書き出し）ができる。実データ `data/tennis_multiview/processed/meiji_4cam/projects.json`（v1、69 clip）は**変更せず**、コピーを一時ディレクトリで読み込む検証をする。
4. 複数クリップで入力・保存・再読み込み・修正（一括再計算を含む）の一連の操作を、ブラウザE2E（`tests/e2e/tennis_scene/clip_studio_browser.mjs` を拡張または新規）で確認する。検証専用datasetは実dataset名を拒否する既存の方針に従い、必要ならworktreeのDATA root配下に合成の短い動画で作る。実行できた場合はコマンドと結果をPRに書き、実行できない場合は理由を書く。
- ユニットテスト: 採点ルール（ソフト: デュース、ファイナルゲーム、サーブ交代／硬式: タイブレーク、ノーアド、マッチタイブレーク）、スキーマv1→v2読込とv2往復・不正値の拒否、連鎖の自動入力・停止・not_point通過・一括再計算の範囲・不連続判定、CSVの全件検証と往復、service経由のUndo/Redo・revision。tests/README.md のテスト作成方針に従って設計する。
- README（`src/tennis_scene/clip_studio/README.md`）を更新（二重管理しない）。

## 進め方
- 検証とpre-commitを通してcommit・pushし、ドラフトでないPRを作成する。PR本文に受入条件ごとの証拠、テストの増分と実測、未実施事項を書く。
- push後はCI待ちを `PARENT_RESULT` の `kind: ci` で宣言して終了する。CI成功後の追加作業はない（`resume_on_success: false`）。
- 長時間のpytest・mypyは `parent.local_jobs start` で起動して `kind: local` で宣言する。
