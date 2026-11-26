# テニスシーン3D再構成のアーキテクチャー

同期固定カメラ映像からのテニスシーン再構成を扱う日本語の技術報告です。章立て・主張・出典候補は [設計書](design.md)、制作手順は [paper-writing スキル](../../.agents/skills/paper-writing/SKILL.md) を参照してください。

## 原稿の構成

| ファイル | 用途 |
| --- | --- |
| [main.tex](main.tex)、[preamble.tex](preamble.tex) | 表題・要旨と原稿の接続、共通の組版設定 |
| [chapters/](chapters/)、[appendix/](appendix/) | 本文と付録。各ファイルの見出しに続けて本文を記入 |
| [figures/](figures/)、[tables/](tables/) | 掲載図版と生成した LaTeX 表 |
| [evidence/](evidence/) | [主張と数値の出典の対応表](evidence/claims.md)、図表の来歴、最終版のビルド記録 |
| [references.bib](references.bib) | 一次資料で確認した書誌情報 |

初めて本文に `\cite{...}` を使うときは、対応する書誌を `references.bib` に登録し、`main.tex` 末尾の参考文献用の二行を有効にします。図表のパスはこのディレクトリを基準に指定します。

## PDF をビルド

LuaLaTeX、LuaTeX-ja、latexmk、BibTeX、`pdfinfo`（Poppler）を使用します。雛形は TeX Live 2023/Debian（LuaHBTeX 1.17.0、LuaTeX-ja 20231230.0）、latexmk 4.83、Poppler 24.02.0 で確認しました。フォントは Noto Serif/Sans CJK JP、DejaVu Serif/Sans です。[コート検出報告の組版手順](../court_robustness/README.md#pdfを再ビルド)

リポジトリのルートで実行します。

```bash
bash paper/tennis_scene_architecture/build.sh
```

PDF と最終ログは `paper/tennis_scene_architecture/build/main.pdf` と `build/main.log` に出力します。スクリプトは latexmk で参考文献と相互参照を解決し、最終ログに未解決の参照・引用、文字欠落、はみ出しがある場合は失敗します。成功時は PDF のページ数を表示します。雛形は見出しだけで組版でき、実験データや参考にした DINOv3 のソースの配置を要しません。

最終版の組版と目視確認は [制作手順](../../.agents/skills/paper-writing/SKILL.md#5-ビルドと最終確認) に従い、`evidence/build.md` に記録します。

## Gitへの登録

worktree が共有する `.git/info/exclude` に `/paper` があるため、追跡する新規ファイルは個別のパスを指定して `git add -f` で登録します。以下は図版・表・ビルド記録を追加した場合の例です。

```bash
git add -f -- paper/tennis_scene_architecture/figures/overview.png \
    paper/tennis_scene_architecture/tables/metrics.tex \
    paper/tennis_scene_architecture/evidence/build.md
git diff --cached --name-only
```

ステージ済み一覧で、その差分に必要な図表・根拠資料・生成スクリプトが揃い、`build/` の中間生成物と PDF が含まれていないことを確認します。

登録後は以下のコマンドで組版し、latexmk が列挙したリポジトリ内の入力を Git インデックスと照合できます。Git と `realpath` を使います。未登録の入力があれば、パスを表示して失敗します。

```bash
bash paper/tennis_scene_architecture/build.sh --check-tracked
```
