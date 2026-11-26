# #2596 担当: 人物モデル比較（#2492）の結果を GitHub Wiki に網羅的にまとめる

あなたは親 #2492（人物モデル比較）の子 #2596 の担当です。ユーザーの依頼（2026-11-26）:
「今回の結果をWikiにまとめたいと思います。Workerに委託し、網羅的な情報をWikiに記述するように頼んでください。」

ページの構成と置き場所は親が決めました（下記）。あなたは事実を main の記録・コード・Issue/PR で確かめて執筆し、検証してから親のレビューを受けます。push するのは承認の後です。

## 前提と禁止事項

- Wiki は `https://github.com/Motoki0705/tennis-lab.wiki.git`（**公開**）。本体repoのファイルは変更・commit・PR 作成しない。Issue コメントも書かない。
- Wiki の clone は、この worktree の `tmp/wiki/` に置く（`tmp/` は gitignore 済み）。
- **公開ページに書かないもの**:
  - ローカル絶対パス（`[local path]`・`[local path]`・`[local path]`・`D:` 等）と、ローカル出力の場所（`.worktree-outputs/...`・`.training_queue/...`・`outputs/...`）。
  - セッションID、queue の job ID・job 名、worker の run ID。
  - ユーザー名・メールアドレス・秘密情報、動画の収録日。
  - 動画フレーム画像、切り抜き、overlay 画像、同期比較動画（映像に人物が写るため）。ローカルにある成果物は「ローカル保管、Wiki には載せない」とだけ書く。
- **図**: Mermaid とテキスト表を使う。main にある比較図の PNG（`pareto-detection.png`・`pareto-detection_and_pose.png`・`pareto-batch1.png`）は、実際に開いて写真や映像の切り抜きを含まないグラフであることを確かめてから、main への URL（`https://github.com/Motoki0705/tennis-lab/blob/main/<path>?raw=true`）で埋め込んでよい。Wiki repo に画像を追加しない。
- claude.ai の統合報告ページは非公開なのでリンクしない。
- データは読み取り専用。GPU と training queue は使わない。
- **推測で書かない。** 数値・条件名・ファイル名は main の記録（CSV・JSON・knowledge ノード・README）かコードで確かめる。確かめられないことは「未確認」と書く。数値は記録の値を丸めるだけにし、新しく計算した値（比・差）には何から計算したかを書く。

## 正本（main の記録）

最新の origin/main を使う。少なくとも次を読む。

- 親 #2492 の本文とコメント（ユーザー判断・合意・方針）、子 #2487・#2490・#2491・#2495・#2496 の本文、マージ済み PR #2497・#2498・#2499・#2500・#2501・#2534・#2573・#2541・#2580・#2593（#2593 が未マージなら、FastHMR の判断は #2494 の親の記録コメントを出典にする）。
- 共通基盤: `knowledge/nodes/tennis_scene/000084-run-meiji-multiview-player-evaluation.md`、`000085-run-person-model-speed-baseline.md`、`000086-run-person-model-speed-low-precision.md` と各 run ディレクトリ。採点コードは `src/tennis_scene/evaluation/player_pose/`。
- 人物検出・統合モデル: `knowledge/nodes/player_detection/000009`〜`000011`、`knowledge/runs/run-meiji-player-model-comparison/`（`conditions.csv`・`README.md`・`EDGECRAFTER.md`・`rfdetr.md`・`rtdetrv4.md`・各 JSON）、`run-meiji-rfdetr-large-accuracy/`・`run-meiji-rfdetr-large-speed/`。
- 2D pose: `knowledge/nodes/tennis_scene/000091-run-meiji-topdown-pose-comparison.md`、`knowledge/runs/run-meiji-topdown-pose-comparison/`（`comparison.csv`・`accuracy.csv`・`difficulty_pt32.csv`・`README.md`・`PMPose.md`・`rtmpose.md`・`sapiens2.md`・各 JSON）。
- 人体メッシュ: `knowledge/nodes/tennis_scene/000092`〜`000095` と各 run ディレクトリ。
- 実装: ECPose 経路と FastHMR 経路の recipe・component（main のコードで、選び方・出力契約・既定 recipe との分離・cache の照合を確かめる）。
- 既存の Wiki ページ: `Meiji-4cam-Benchmarks`（ベンチマークの説明はここにあるので重複させずリンクする）、`Component-Person-Detection-and-Tracking`、`Component-3D-Reconstruction-and-Tooling`、`Tennis-Scene-History-and-Optimizations`。

## 親が決めたページ設計

### 置き場所

「I. `tennis_scene` 標準パイプライン」に新規5ページを置く。slug と見出しは次のとおり（見出しの細部は調整してよい）。

| slug | 見出し |
|---|---|
| `Person-Model-Comparison-Overview` | 人物モデル比較（#2492）概要・結論 |
| `Person-Model-Comparison-Methods` | 人物モデル比較の方法（疑似GT・採点・速度計測・出力差） |
| `Person-Detector-and-ECPose-Comparison` | 人物検出・統合モデル（ECPose）の比較 |
| `Top-Down-Pose-Model-Comparison` | Top-down 2D pose の比較 |
| `Body-Recovery-FastHMR-vs-GVHMR` | 人体メッシュ再構成（FastHMR と GVHMR）の比較 |

既存ページの変更は次だけ（各1〜数行）:

- `_Sidebar.md` の I 節に、5ページへの導線（`人物検出・2D姿勢・追跡・選別` の後に、小見出しか字下げで）。
- `Home.md` の I 行: ページ数を実数に合わせ、主な内容に人物モデルの代替比較を一言足す。
- `Tennis-Scene-Pipeline-Overview.md`（I の入口）のページ一覧に5行。既存の書式に合わせる。
- `Component-Person-Detection-and-Tracking.md`・`Component-3D-Reconstruction-and-Tooling.md`・`Tennis-Scene-History-and-Optimizations.md`・`Meiji-4cam-Benchmarks.md` に、関係するページへの相互リンクを1行ずつ。
- 他のページは変更しない。

### 書式

既存ページに合わせる: 冒頭の表（想定読者・検索キーワード・正本ファイル）、入口ページへの戻りリンク、`[表示テキスト](Slug)` の内部リンク、GitHub の `> [!NOTE]` 等。本体 repo へのリンクは `https://github.com/Motoki0705/tennis-lab/blob/main/...` で、**main に存在するファイルだけ**に張る（`git ls-tree origin/main` で確認）。各ページに「基準日 2026-11-26」を書く。日本語。分量の目安は概要 120〜200 行、方法 150〜250 行、検出 200〜320 行、pose 150〜260 行、人体メッシュ 100〜180 行。網羅を優先し、同じ事柄を2ページに重複して書かない（詳細は1か所に書き、他はリンクする）。

### 各ページの必須内容

**1. 概要・結論（`Person-Model-Comparison-Overview`）**

- 目的と背景（現行 DINO の処理時間の比率についての問題認識と、計測で確かめた範囲）、スコープ、子 issue と PR の対応、進め方（共通基盤を先に作り、各段を並行して比較）を Mermaid で。
- スコープを変えたユーザー判断の一覧（例: Sapiens2・RF-DETR-2XL の除外、PT32 は計測せず「PT32未計測」と表示、出力差は共通の基準で判定して並記、FastHMR の PT16・TRT16 は速度だけの参考値、PT32 の重みからの ONNX 変換は可、FastHMR の置き換え見送り）。それぞれの日付と出典。
- 段ごとの結論表: 現行モデル、最良候補、速度（現行 → 候補）、精度（現行 → 候補）、PT32 との出力差、判断の状態。経路全体のトレードオフの説明。経路全体の速度は段ごとの計測からの見込みで、全体を通しで計測していないことを明記する。
- 親の推奨（採用前の候補）と、その根拠・弱点（例: RF-DETR-L の Recall@0.75 の低下と共通 parity の不成立）。代替候補。不採用・見送りの候補と理由。
- 状態: 既定モデル・既定 recipe は変えていない（切替は別の承認事項）。#2487 の下流確認（候補を標準経路につないで追跡と3Dへの影響を見る）は未完了で、ユーザーの選択待ち。
- 残っている制約・未確認の点の一覧。

**2. 方法（`Person-Model-Comparison-Methods`）**

- 共通ベンチマーク: 代表6 clip と選定理由、評価フレーム・カメラ・画像数。詳細は `Meiji-4cam-Benchmarks` にリンクし、ここでは比較に効く点だけ書く。
- 疑似GT: GVHMR → SMPL → 配置 → COCO17 → 2D 射影の作り方、疑似 BBox の規則、品質 mask・理由コード、母数・対象外の扱い。**何が GT でないか**（三角測量 COCO17 と ViTPose は GT に使わない、疑似GTは真値ではない）。**循環性**: 疑似GTが現行経路（DINO・ViTPose・GVHMR）の出力に由来するなら、現行モデルに有利な偏りがありうることを、実際の生成経路をコードと記録で確かめたうえで説明する。
- 採点指標: 検出の Recall@0.50・Recall@0.75（対応付けの方法、未ラベル人物の扱い）、pose の OKS（sigma・面積の定義）、統合モデルの採点。main の採点コードで確かめる。
- 速度計測: 共通 harness（#2497）の計時境界（何を含み何を含まないか）、batch、CPU スレッド数、warmup・反復数、P50・P95、VRAM の測り方（NVML と allocator の違い）、GPU・ドライバ・CUDA・TensorRT の版（記録にあるもの）。PT32・PT16・TRT16 の意味と、PT32 を測らない方針。
- PT32 との出力差（parity）: 共通の基準（検出・pose・人体メッシュそれぞれ）と、判定結果を図から外さず並記する方針。旧基準で合格だった記録の扱い。
- 再現: main にある再現スクリプトへのリンク。

**3. 人物検出・統合モデル（`Person-Detector-and-ECPose-Comparison`）**

- 候補一覧: モデル、重み・学習データ、公開元、ライセンス（記録にある範囲。未整理なら「未整理」）、PT16・TRT16 の可否。
- `conditions.csv` の全条件の表（実測値、実行不能・除外の理由）。精度（Recall@0.50・Recall@0.75 など記録にある指標）、速度、VRAM、PT32 との出力差。表が長ければモデル群ごとに分ける。
- 比較図（Pareto）の埋め込みと読み方。
- 統合モデル（ECPose L・X）の検出と pose の結果、検出＋pose の合計時間、不採用の理由。
- ECDet の TRT16 で精度が大きく落ちた件の調査経過と結論（原因が特定できたか、FP32 attention などの切り分け、図からの扱い）。
- 実装: ECPose 経路を明示選択する方法、出力契約、既定 recipe と cache を混ぜない仕組み（#2499 のレビュー対応を含む）。
- 失敗例・小さい人物・遮蔽など、記録にある傾向。

**4. Top-down 2D pose（`Top-Down-Pose-Model-Comparison`）**

- 候補一覧（ViTPose-H の flip あり・なし、RTMPose、PMPose、Sapiens2 など記録にあるもの）、重み・ライセンス、除外の理由（Sapiens2 はユーザー判断と、5B の OOM 等の記録）。
- `comparison.csv` の全条件の表: OKS、batch 1・4・8 の速度（ms/人）、VRAM、PT32 との出力差、実行不能の理由。
- 難しい条件ごとの精度（`difficulty_pt32.csv` 等、記録にあるもの）。
- 比較図の埋め込み。
- RTMPose-X TRT16 を候補にする根拠と、採用前に残る課題（confidence の較正、重み・engine の利用条件など記録にあるもの）。

**5. 人体メッシュ（`Body-Recovery-FastHMR-vs-GVHMR`）**

- 構成の比較: 現行（HMR2 特徴＋GVHMR）と FastHMR（CameraHMR）の重み・commit・入出力契約・ライセンス（CameraHMR の重みは登録制の配布であることなど、記録にある範囲）。
- 全条件の表: PT16・TRT16 の速度（batch 1・2・4、ms/系列と ms/人・フレーム）、VRAM、実行不能の理由（TRT16 書き出しの停止理由）。PT32 の8フレーム試行の関節差。
- 6 clip の同期比較動画を作ったこと（ローカル保管、Wiki 非掲載）と、目視評価は行っていないこと。
- ユーザー判断（2026-11-26）: 速度の優位がないため置き換えは見送り。既定は GVHMR のまま。`recipe=fasthmr` は明示選択の経路として残り、その選び方。

## 手順

1. `git clone https://github.com/Motoki0705/tennis-lab.wiki.git tmp/wiki` し、既存ページの書式を読む。
2. 上の事実を main の記録・コード・Issue/PR で確かめる。確かめたコマンドと結果の要点、各表の数値の出典（ファイルと行・キー）を `tmp/wiki-evidence.md` に残す（公開しない）。
3. 5ページを書き、既存ページの導線を直す（`tmp/wiki/` の中で編集する。commit・push はまだしない）。
4. 検証する:
   - 各表の数値を、出典の CSV・JSON ともう一度照合する（できればスクリプトで突き合わせ、その結果を evidence に残す）。
   - 内部リンクの slug がすべて Wiki に実在し、本体 repo へのリンク先が origin/main に実在する。
   - 禁止事項の検索（`[local path]`・`[local path]`・`[local path]`・`.worktree-outputs`・`.training_queue`・`@`・メールアドレス・セッションID や job ID の形）で該当なし。
   - Mermaid が既存ページと同じ書き方で、日本語ラベルを引用符で囲んでいる。
5. **ここで止まる。** commit も push もしない。変更ファイル一覧、各ページの節構成と行数、各表の出典、「未確認」「未整理」として残した項目、判断に迷った点を最終回答にまとめ、`PARENT_RESULT` を `needs_parent` で返す。親が `tmp/wiki/` の内容を読んでレビューする。
6. 親から push の承認と修正指示が来たら、修正し、`git pull --rebase` してから Wiki に commit・push する（commit メッセージは日本語で既存に合わせる。`Co-Authored-By` 等は付けない）。push 後に5ページの公開 URL が開けることを確かめ、commit hash を報告して `done` を返す。

## 最終回答の末尾

最終行に1行 JSON を置く。

- レビュー待ちで止まるとき: `PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}`
- push まで終えたとき: `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
