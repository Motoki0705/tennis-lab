# Wiki アセット生成ツール

本体リポジトリの固定版 CSV／JSON から Wiki の数値表とグラフを生成する。ページの定義は [pages/](pages/) にあり、表・グラフ・折りたたみテキスト表が同じデータを参照する。執筆と配置の規約は [GitHub Wiki 執筆ガイドライン](../../docs/wiki/README.md) を参照する。

## 実行環境

- 本体の `.venv` の Python、Matplotlib、Pillow、markdown-it-py、filelock を使用する。描画は CPU で実行する。
- 日本語に Noto Sans CJK JP、表の数字に DejaVu Sans Mono を使う。生成環境に両フォントの regular／bold を用意する。例えば Ubuntu では `fonts-noto-cjk` と `fonts-dejavu-core` を導入する。
- Typst は [公式の 0.15.1 リリース](https://github.com/typst/typst/releases/tag/v0.15.1) を使う。x86-64 Linux では、既定で公式アーカイブと取り出した実行ファイルの SHA-256 を検査し、`$XDG_CACHE_HOME/tennis-lab/wiki-assets/`（未設定時はキャッシュホーム直下の `tennis-lab/wiki-assets/`）に保存する。`--typst-bin` または `WIKI_ASSETS_TYPST_BIN` で手持ちの 0.15.1 を指定できる。引数が環境変数より優先される。`--offline` はキャッシュまたは明示パスを必須にする。

## 生成と公開前検証

本体リポジトリのルートから実行する。

```bash
.venv/bin/python -m tools.wiki_assets list-pages
.venv/bin/python -m tools.wiki_assets generate --output tmp/wiki-assets-preview
.venv/bin/python -m tools.wiki_assets verify --output tmp/wiki-assets-preview
```

`--page <page-slug>` を指定するとそのページだけを対象にできる。複数ページを指定する場合はオプションを繰り返す。既定の正本は [入力の固定版](sources.py) にある。比較結果を更新する場合は、公開済みの Git リビジョンを生成・検証の両方に `--data-revision` で渡す。入力は Git から直接読む。対象リビジョンが取得済みの checkout を使う。

- `assets/<page-slug>/` に SVG と PNG が出力される。SVG と PNG はそれぞれ 2 MiB 以下、PNG は各辺 4,096px 以下。生成先には対象ページ専用のディレクトリを使う。
- `<page-slug>.md` に埋め込みスニペットが出力される。Wiki のページ本文に必要な節を取り込み、対応する画像を Wiki の `assets/` 以下へ置く。
- `manifests/<page-slug>/manifest.json` に入力 SHA-256、行ごとの元値と表示値、除外理由、順位、出力 SHA-256、描画ツール版が出力される。生成時に検査した後、対象ページの出力を置き換える。別の生成処理と同じ出力先を共有するときは、各処理が完了してから検証する。

`verify` は指定版から値、状態、順位、テキスト参照を再計算し、manifest と Markdown、画像の SHA-256 と構造を照合する。同じツール・フォントで画像を再生成して比較する場合は `--rerender` を付ける。フォントや描画ライブラリの版を変えると画像のバイト列も変わり得る。SVG／PNG の文字などは表示を目視する。出力ファイルへの機密情報混入は生成時にも検査する。

Wiki ページへ図表を移すときは、生成スニペットの各図表を `<!-- wiki-assets:begin … -->` から `<!-- wiki-assets:end … -->` までコピーする。使うブロックを選び、ブロック間に空行と手書きの見出し・本文を置ける。各マーカーは同じページに一度だけ置き、マーカー間の値・単位・状態・リンクを保持する。マーカーがない旧スニペットは再生成する。

公開前は検証済みの出力と Wiki 実ファイルを同じコマンドに渡す。マーカーのページ slug とアセット名で、コピーしたブロックの全文と、Wiki に置いた SVG／PNG の SHA-256 を照合する。選択したバンドルの画像には対応するマーカー付きブロックが必要。複数の Wiki ファイルは `--wiki-page` を繰り返して指定できる。

```bash
.venv/bin/python -m tools.wiki_assets verify --output tmp/wiki-assets-preview \
  --page person-model-comparison-overview \
  --wiki-page ../tennis-lab.wiki/Person-Model-Comparison-Overview.md
```

`--wiki-page` 単独では、Markdown の書式、相対リンク、参照先の SVG／PNG の形式を検査する。アセットのパスは Markdown の所在を基準に解決する。

## テスト

```bash
.venv/bin/python -m pytest -n 0 tests/unit/tools/test_wiki_assets
.venv/bin/ruff check tools/wiki_assets tests/unit/tools/test_wiki_assets
.venv/bin/mypy tools/wiki_assets tests/unit/tools/test_wiki_assets
```

テストはネットワーク接続なしで実行できる。実 CLI の往復テストには `WIKI_ASSETS_TYPST_BIN` で 0.15.1 を指定する。未指定時は、オフラインでバイナリが見つからない場合のエラー契約と、Matplotlib・公開形式・数値照合の検証が動く。
