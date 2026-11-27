# #2602 子Issue 担当: `Meiji-4cam-Benchmarks` の階層型ページ改訂と公開前検証

#2694 で確定した情報階層化方針と冒頭定型メタデータ欄廃止方針の最初の実例として、`Meiji-4cam-Benchmarks.md` を階層型構成へ改訂し、本番 Wiki 公開前の全検査（情報欠落・リンク・実表示・公開禁止情報）を完了してください。

## 厳守事項・境界
- **本番 GitHub Wiki への `git push` はまだ行わないでください！** 本番公開は親レビューおよび人間確認ゲート（Tier 1 代表ページ確認）の承認後に行います。
- **本体リポジトリのファイルは変更・commit・PR 作成しないでください。** Issue コメント投稿も行わず、親への報告は最終メッセージと worktree 内 `tmp/meiji-revision/` の成果物にまとめてください。
- GitHub Wiki（`https://github.com/Motoki0705/tennis-lab.wiki.git`）はこの worktree の `tmp/wiki/` に clone し、`tmp/wiki/Meiji-4cam-Benchmarks.md` に改訂ドラフトを作成してください（他の既存 Wiki ページ・`_Sidebar.md`・`_Footer.md` は変更しない）。

## 参照する先行設計（#2694 の成果物、読み取り専用）
- `[local path]`（「Meiji-4cam-Benchmarks の構成」および「旧版との対応」）
- `[local path]`（旧版全 242 行・62 リンクの対照データ）
- `[local path]`

## 改訂要件
1. **冒頭定型メタデータ欄の完全廃止**:
   - 冒頭の「項目／内容」表（「想定読者」「検索キーワード」「正本ファイル」）、定型戻りリンク（`> ← [研究モデル・基盤リファレンス入口へ戻る](...)`）、定型「基準日 2026-11-25」ブロックを削除し、ページタイトル `# meiji_4cam マルチビュー ベンチマーク（multiview-v1/v2・人物評価v1）` の直後から本題に入れる構成にする。
   - ただし、旧冒頭にあった必要な情報（各版の `README.md`・`manifest.json`、人物評価コードの仕様へのリンク、`Person-Model-Comparison-Overview`・`Person-Model-Comparison-Methods` への導線、および文書の版参照時点としての `2026-11-25`）は該当本文および「出典と関連 Issue・PR」節へ確実に維持する。
2. **全体像と系譜の階層化**:
   - 4 版（ローカル `multiview-v1` / `multiview-v2` / `multiview-player-evaluation-v1` と派生配布の `Hugging Face 配布版`）の関係・用途・容量の目安を、一目で全体像が掴める **ディレクトリツリー形式** でタイトル直後の「概要と系譜」に配置する。
   - 派生・依存関係を狭幅でも潰れにくい **縦方向 Mermaid 図（`flowchart TD`）** で表現し、図の直後に短い説明と元 Issue リンクを添える。
   - 共通のローカル起点と読み取り専用原則（`> [!NOTE]`）を維持する。
3. **`Hugging Face 配布版` セクションの構造化**:
   - 既存見出し `## Hugging Face 配布版`（アンカー `#hugging-face-配布版`）を維持し、さらに「収録物」「取得と検算」「利用方法」の階層で整理する。
   - 閲覧権限、SMPL メッシュはローカル `multiview-v2` を使う点、固定 revision `e919bcba733543a83d92b8fbb379fa41a6d36978`、`hf download` + `sha256sum -c SHA256SUMS`、`index.jsonl`・`clips/clip_XXX/media/camN.mp4`・`scene.npz`・`scene.metadata.json`・`load_scene_result()`、データセットカードと SceneResult 仕様へのリンクを完全維持する。
4. **ディレクトリ・ファイル仕様の折りたたみ式階層化**:
   - 既存見出し `### multiview-v1`、`### multiview-v2`、`### multiview-player-evaluation-v1` を維持する。
   - 各版の短い用途を折りたたみの手前に置き、`<details><summary>...<[local path]>` ＋ **入れ子リスト（重要：`<details>` / `<[local path]>` / `<[local path]>` と Markdown リスト・コードブロックの前後に必ず空行を入れる）** でディレクトリ・ファイルごとの役割を構造化する。
   - コピー用テキストツリー（` ```text ... ``` `）も各版内で開けるように維持する。旧版の 2 列表（ファイル・部品／内容）にあった全記述を対応する階層ノードへ欠落なく移設する。
5. **データ仕様（収録・座標・配列仕様・既知の問題）の階層化**:
   - `SceneResult v2の配列` と `人物評価v1の疑似GTと固定入力` の長文 2 列表を、主体別・ファイル別の入れ子リストへ整理する（軸 `P, C, T, J`、形状、マスク・棄却コード、無効値=0 / NaN / `segment_index=-1`、疑似GTの注意 `> [!IMPORTANT]` を完全維持）。
   - `整合性と既知の問題` の全数値・全クリップ番号・全注意点を維持する。
6. **`各モデルでの利用可否` と `考えられる用途` の階層化**:
   - 旧版で通常幅高さ 2,160px・狭幅 3,888px（横幅 724px）だった巨大な 4 列表（10 経路）を、「元映像・既存固定入力から使える経路」と「配列・幾何をモデルの契約へ変換する経路」の **階層型リスト** に再構成し、各経路の下に「入口・変換」「評価・学習」「注意点・出典」を整理する。全 10 経路のすべてのリンクと注意点を維持する。
   - `学習データとの重複`（`> [!WARNING]` および各経路の照合済み・未照合範囲と出典リンク）を維持・見やすく整理する。
   - `考えられる用途`（6 用途）も表をやめて用途別の簡潔な階層リストへ整理する。
   - 最終的にこのページには**横幅過多な長文 Markdown 表が 0 件**になるようにする（残す表がある場合は 2〜4 列・短セルの基準を満たすもののみ、原則ツリーと階層リストで統一する）。

## 必須の公開前検証（成果物: `tmp/meiji-revision/`）
ドラフト完成後、以下の検証スクリプト・記録を `tmp/meiji-revision/` に作成して全件確認してください：
1. **情報欠落・リンク突合検査 (`tmp/meiji-revision/migration-audit.md` / `.json`)**:
   - 旧版 `Meiji-4cam-Benchmarks.md` の全 242 行、全 62 リンク（外部 URL・Wiki 相対リンク・アンカー）、全コードブロック、全数値・クリップID・SHA/revision・ファイル名・配列名・注意点について、新版での対応箇所を 1 対 1 で突合し、欠落 0 件を確認する。
   - すべての外部 GitHub URL / Wiki 相対リンクの整合性を確認する。
2. **書式・構文・公開禁止情報検査 (`tmp/meiji-revision/preflight.json`)**:
   - 冒頭定型メタデータ表・定型戻り行・定型基準日ブロックが存在しないこと。
   - すべての `<details>` / `<summary>` / `<[local path]>` タグの開閉整合と、内包する Markdown ブロック前後の空行ルールが守られていること。
   - Mermaid 構文エラーがないこと。
   - 公開禁止情報（ローカル絶対パス、個人情報、トークン等）が 0 件であること。
3. **ローカル実表示検証 (`tmp/meiji-revision/screenshots/` & `render-check.md`)**:
   - GitHub 相当の Markdown HTML レンダリング（`markdown-it-py` + GitHub スタイル幅 896px / 390px、ライト `#ffffff` / ダーク `#0d1117`、`<details>` 開閉状態）を Chromium で描画・スクリーンショット採取し、変更前（旧版）と変更後（新版）の高さ・必要横幅・390px での横スクロール有無・可読性を比較記録する。

## 報告・終了形式
- 最終メッセージに、改訂ドラフトのパス（`tmp/wiki/Meiji-4cam-Benchmarks.md`）、旧版→新版の変更概要（行数・表数・高さ・横スクロール改善）、全 62 リンク＆情報欠落突合結果、書式・公開禁止情報検査結果、ローカル表示比較結果を簡潔にまとめ、末尾に次の 1 行を出力してください：
`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
