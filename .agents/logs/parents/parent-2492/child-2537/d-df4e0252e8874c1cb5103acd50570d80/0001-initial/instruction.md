# #2492 担当 wiki: meiji_4cam ベンチマークを GitHub Wiki に1ページ追加する

あなたは親 #2492（人物モデル比較）の子 #2537 の担当です。ユーザーの依頼（2026-11-25）:
「今回作成したベンチマークのデータセットについてWIKIに追加してほしい。どのようなディレクトリ構成かを明確にし、データセットの仕様、SLCSなどで使用可能か、どのような用途が考えられるかを書く。」

ページの構成と置き場所は親が決めました（下記）。あなたは事実を実データ・コードで確かめて執筆し、検証してから親のレビューを受け、承認後に Wiki へ push します。

## 前提と禁止事項

- Wiki は `https://github.com/Motoki0705/tennis-lab.wiki.git`（**公開**）。本体repoのファイルは変更・commit・PR作成しない。Issueコメントも書かない。
- Wiki の clone は、この worktree の `tmp/wiki/` に置く（`tmp/` は gitignore 済み）。
- **公開ページに書かないもの**: ローカル絶対パス（`[local path]`・`[local path]`・`[local path]`・`D:` 等）、セッションID・queue job ID、ユーザー名・メールアドレス、秘密情報。動画のフレーム画像・切り抜きは載せない（映像に人物が写るため）。図は Mermaid とテキストだけ。
- データのパスは repo 相対（`data/tennis_multiview/processed/meiji_4cam/benchmarks/...`）で書き、データ本体は git repo にも Wiki にも含まれずローカル保管であることを明記する。
- データは読み取り専用。ベンチマーク配下のファイルを書き換えない・追加しない。GPU・training queue は使わない。
- 推測で書かない。数値・ファイル名・配列名は実ファイル（`manifest.json`・`README.md`・`SHA256SUMS`・実際の `find`[local path]`np.load` の結果）かコードで確かめる。確かめられないことは「未確認」と書く。

## 対象データ（`data/tennis_multiview/processed/meiji_4cam/benchmarks/`）

1. `multiview-v1`: #1739 の新経路で再構成した meiji_4cam `video_000` の全69 clip（4カメラ、scene・exports・completion）。#2446（PR #2470）。
2. `multiview-v2`: v1 のうち clip_012・015・026 のボールを再注釈版 rev1 で再構成した版。人物の配列は v1 と同一。現在の推奨版。
3. `multiview-player-evaluation-v1`: v1 から選んだ6 clip（6,844枚）の人物評価用 bundle。疑似GT・品質集計・投影監査・身体復元の固定入力。#2495。採点コードは main の `src/tennis_scene/evaluation/player_pose/`。

## 親が決めたページ設計

### 置き場所

- 新規ページ1つ。slug `Meiji-4cam-Benchmarks`、見出し「meiji_4cam マルチビュー ベンチマーク（multiview-v1/v2・人物評価v1）」。目安150〜250行。
- 既存ページの変更は次の3か所だけ（それぞれ1行程度）:
  - `_Sidebar.md` の「VI. 研究モデル・基盤リファレンス」で、`研究用3Dモデル (PLCS/BLCS/SLCS)` の直後に1行。
  - `Research-and-Infra-Overview.md` のページ一覧表に1行（既存行の書式に合わせ、番号の振り直しが必要なら最小限に）。
  - `Home.md` の VI 行のページ数（「5 ページ」）を実数に合わせる。説明文に一言足すのは可。
- 他ページは変更しない。

### 書式

既存ページ（例 `Research-Models-PLCS-BLCS-SLCS.md`）に合わせる: 冒頭の表（想定読者・検索キーワード・正本ファイル）、入口ページへの戻りリンク、`[表示テキスト](Slug)` の内部リンク、GitHub の `> [!NOTE]` 等。本体repoへのリンクは `https://github.com/Motoki0705/tennis-lab/blob/main/...` で、**main に存在するファイルだけ**に張る（`git ls-tree origin/main` で確認）。未マージのもの（例: 速度計測 harness #2496 / PR #2497）は Issue/PR 番号だけで触れる。ページ内に「基準日 2026-11-25」を書く。日本語。

### 必須の節

1. **概要と系譜**: 3つの版の関係（入力 #2363・#2458・#2445 → v1 → v2 / 人物評価v1）を Mermaid で示す。どの用途にどの版を使うか、読み取り専用で修正は新しい版を作る方針、正本は各版の `README.md` と `manifest.json` であること、容量の目安。
2. **ディレクトリ構成**: 各版のツリーを実データの `find` で確かめて書く（繰り返しは `clip_XXX` や `…` で省略し、件数を添える）。主要ファイルごとに「何が入っているか」を1行で。v1 は clip 配下の `scene/`（`scene.json`・`components/` の各コンポーネント・`exports/`）まで、v2 は v1 との差分（`ball_revision/`・`completion/ballrev1/`・hard link 共有）、人物評価v1 は `clips/<clip>[local path]` 配下の全ファイル種別まで。
3. **データ仕様**:
   - 収録条件: カメラ数・解像度・fps・clip数・フレーム数（総数と範囲）、人物評価v1 の評価画像数と選んだ clip。
   - 座標系とカメラ: コート座標（単位・軸・原点）、`K, R, t` の意味、画素座標の約束。
   - scene の中身: 選手（身体配置・3D関節）とボール（`ball_3d`・`ball_uv`・`ball_vis` 等）の配列、有効性mask・棄却コード。`scene.json` と npz を実際に開いて確かめる。
   - 人物評価v1 の配列: 軸（P,C,T,J）、疑似GTの定義と作り方、品質・理由コード、`body_recovery_inputs.npz`。
   - **何が正解（GT）でないか**: 三角測量した COCO17 と ViTPose は GT に使わない（ユーザー決定）。疑似GTは GVHMR 配置由来で真値ではない。ボール3Dは2D座標の三角測量で、人手注釈は rev1 の3 clip の2D座標だけ。
   - 整合性の確かめ方: `sha256sum -c SHA256SUMS`、manifest sha256 の固定（`load_reference_frames(..., expected_sha256=...)` 等、main のコードで確認）。
   - 既知の問題: 各 manifest の `known_issues` を要約し、正本は manifest と明記。
4. **各モデルでの利用可否**（表）: 行は少なくとも、人物検出、2D姿勢、身体復元（HMR系）、`tennis_scene` 標準パイプラインの回帰確認、PLCS（多視点）、単眼PLCS（#2475、main に入った tennis_scene の単眼PLCS component）、BLCS、SLCS、`ball_refiner_3d`、ボール2D検出。列は「そのまま使えるか／変換が要るか（変換コードが main にあるか、名前）[local path]」。
   - 各行は該当モジュールの README・データ契約・設定を読んで判定する。変換コードがなければ「変換コードなし」と書き、作らない。
   - **リークを必ず調べる**: SLCS の実RGBデータ（`knowledge/reports/slcs-real-rgb.md`・`src/tasks/slcs/configs/train_real_rgb_temporal_domain_balanced.yaml`）では Meiji `video_000` が train split にある。これがこのベンチマークと同じ映像（meiji_4cam `video_000`）かを、データ生成の入力・clip ID・映像 sha256 などで確かめる。同様に、PLCS・BLCS・ボール検出・ball_refiner_3d の既存 checkpoint の学習データに meiji_4cam `video_000` が含まれるかも確認し、含まれるなら「その checkpoint の評価には使えない（学習済み）」と書く。確認できなければ「未確認」。
5. **考えられる用途**: 用途ごとに、使う版・前提・注意を書く。少なくとも、人物モデル（検出・姿勢・身体復元）の精度比較（#2492 の現在の使い方）、既定モデル変更時の回帰確認、単眼・多視点3Dモデルの実データ評価（リーク条件付き）、ボール三角測量・refiner の評価（rev1 の3 clip の扱い）、速度計測の入力、学習データ生成への転用（学習に使った時点で評価用ではなくなる点）。
6. **出典と関連 Issue・PR**: 生成 Issue/PR、ボール再注釈の Issue（manifest の provenance で確認）、#2495 の PR、#2492・#2496、main のコードへのリンク。

## 手順

1. `git clone https://github.com/Motoki0705/tennis-lab.wiki.git tmp/wiki` し、既存ページの書式を読む。
2. 上の事実を実データとコードで確かめる。確かめたコマンドと結果の要点を `tmp/wiki-evidence.md` に残す（公開しない）。
3. ページを書き、既存3ページを直す（`tmp/wiki/` 内で編集。commit・push はまだしない）。
4. 検証する:
   - 内部リンクの slug がすべて Wiki に実在する。本体repoへのリンク先が origin/main に実在する。
   - 禁止事項の grep（`[local path]`・`[local path]`・`[local path]`・`@`・メールアドレス・セッションID の形）で該当なし。
   - Mermaid が既存ページと同じ書き方で、日本語ラベルは引用符で囲んでいる。
   - 数値をもう一度 manifest と照合する。
5. **ここで止まる**。commit も push もせず、変更ファイル一覧・ページの節構成・利用可否表の判定と根拠・未確認として残した項目を最終回答にまとめ、`PARENT_RESULT` を `needs_parent` で返す（下記）。親が `tmp/wiki/` の内容を読んでレビューする。
6. 親から push の承認と修正指示が来たら、修正し、`git pull --rebase` してから Wiki に commit・push する（commit メッセージは日本語で既存に合わせる。末尾に `Co-Authored-By` 等は付けない）。push 後に公開URL（`https://github.com/Motoki0705/tennis-lab/wiki/Meiji-4cam-Benchmarks`）を `gh api` 等で確かめ、commit hash を報告して `done` を返す。

## 最終回答の末尾

最終行に1行JSONを置く。

- レビュー待ちで止まるとき: `PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}`
- push まで終えたとき: `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
