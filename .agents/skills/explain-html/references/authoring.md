# 初版と運用

この文書はskillのhelperとbriefの準備を説明する。文書・根拠・assets/figuresの契約、許可されるHTML/CSS、
グラフ定義、model/effort、HTTP・保存の仕様は [runtime README](../../../ops/explain_html/README.ja.md) を参照する。

## 準備

repo rootで `.venv/bin/python` を使う。Chromiumによる実描画検査にはrepoの既存Playwright依存が必要。
未導入なら既存lockで `npm ci --prefix scripts/ci_metrics` と
`scripts/ci_metrics/node_modules/.bin/playwright install chromium --only-shell` を実行する。
必要に応じて `PLAYWRIGHT_MODULE` と `CHROMIUM_PATH` で導入済みの実体を明示できる。
helperは利用不可のブラウザ検査をpassedにしない。
執筆には現行visual policyに合わせ、`scroll-behavior` 宣言を使わない指示を初版・更新とも渡す。
reduced-motionへの配慮はこの宣言なしで行う。

adapterの既定agyをこの環境で実行できなければ、既存 `agents.conf` のpinと照合した実体を
`create` と `open` の `--agy /absolute/path/to/agy` へ渡す。pinやモデルを変更して回避しない。
出力rootはGit対象外の `outputs/explain-html/<name>/` にし、private記録も含めて保持する。

## brief

helperの `Brief` は `document_id`、`instruction`、runtimeの `WritingContext` を入れる `context`、
asset IDからローカルファイルへの対応 `asset_files`（省略時は空）で構成する。
`asset_files` の相対パスはbriefファイルの親から解決する。public Asset.pathとは別の入力パスである。
型は `scripts/explain_html_author.py` とruntimeからimportしてJSONを作る。schemaの手書きコピーは不要。

以下は図なしの最小入口。実際には根拠・条件と、題材に応じた説明設計を入れる。

```python
import json
import sys
from pathlib import Path

sys.path[:0] = [".agents/ops", ".agents/skills/explain-html/scripts"]
from explain_html_author import Brief
from explain_html.editor import WritingContext

output = Path("outputs/explain-html/example")
output.mkdir(parents=True, exist_ok=True)
brief = Brief(
    document_id="example",
    instruction="用語を導入し、全体像から理由・具体例へ進む日本語の説明を書いてください。",
    context=WritingContext(
        purpose="同じ時刻の複数視点がなぜ必要かを理解する。",
        audience="画像座標は知っているが3D復元は初めての読者。",
        evidence=[],  # 調べたEvidenceをここへ入れる
        preferences="必要な理由を本文へ。補足はdetailsへ。狭幅・keyboard・contrastを確認できるHTML。"
        "保存版が閲覧専用である旨を小さなfooterへ。",
    ),
)
(output / "brief.json").write_text(brief.model_dump_json(indent=2), encoding="utf-8")
```

EvidenceのsourceがHTTP(S)なら本文から実リンクを張り、`data-evidence-id` の範囲に値と条件を保つ。
許可されていない根拠の改変はruntimeが拒否する。読み手が理解するのに足りない資料は補うか、不明と明記する。

## 初版用の画像・図を入れる

imagegenで生成したファイルを、まずbrief隣の `assets/` などへコピーして画像を確認する。
同じPython準備処理で次のように実bytesからAssetを作る。imagegenを呼ぶCLIへ置き換えない。

```python
from explain_html.images import import_image
from explain_html.schema import Figure, Provenance

imported = import_image(
    output / "assets/concept.png",
    asset_id="concept",
    caption="場面を示すAI生成の概念イラスト。寸法や角度は実測ではない。",
    alt="異なる位置の2台のカメラが同じ対象を見る場面。",
    provenance=Provenance(kind="generated_concept", source="Codex interactive imagegen; generation date and purpose"),
)
figure = Figure(
    figure_id="scene-image", content_id="scene", kind="image",
    asset_ids=[imported.asset.asset_id], caption=imported.asset.caption,
    alt=imported.asset.alt, provenance=imported.asset.provenance,
)
context = brief.context.model_copy(update={"assets": [imported.asset], "figures": [figure]})
brief = brief.model_copy(update={"context": context, "asset_files": {"concept": "assets/concept.png"}})
(output / "brief.json").write_text(brief.model_dump_json(indent=2), encoding="utf-8")
```

Geminiへ `scene` block内に空の `<figure data-figure-id="scene-image"></figure>` を置く指示を渡す。
SVGも同様に `visuals.import_svg` のAsset/bytesとFigureを用意する。
グラフはruntimeの定義モデルを使い、`managed_renderer()` のasset ID/versionをFigureに指定する。
helperは準備されたグラフがある場合、現行managed rendererのmetadataと実bytesを追加する。
初版用に自分で用意したrenderer/JSへ差し替える入口はない。

## 生成・起動・再open・停止・export

```sh
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py create --root outputs/explain-html/example/runtime --brief outputs/explain-html/example/brief.json
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py open --root outputs/explain-html/example/runtime --document example --browser
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py status --root outputs/explain-html/example/runtime
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py stop --root outputs/explain-html/example/runtime
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py export --root outputs/explain-html/example/runtime --document example
```

`create` は既存documentを上書きしない。Geminiの執筆/独立意味検査→最終候補のstage→実bytesのvisual/ブラウザ
検査→commitを行い、文書IDと版IDを返す。`open` は既存文書だけを開き、生成を再実行しない。
serviceが動作中ならそのsessionを再利用し、停止中なら固定UI・同じ描画validatorを指定して起動する。
`--browser` は既定ブラウザで開こうとし、結果を `browser_opened` で返す。WSLなどで開けなければ返されたURLを
Windows Edge等へ渡す。通常のリロードではtokenを失うため、再度 `open` のURLを使う。
URLはsessionの操作権限を含むので、証拠のスクリーンショット・動画・共有資料にはfragmentを残さない。

`export` はserviceが停止中でも使える。返されたdirectoryの一式を保存し、`index.html` を開く。
`--revision` で公開済みの旧版も選べる。画像と対応グラフは同梱され、編集UIやprivateログは含まれない。
停止・復旧・公開HTTPの詳細はruntime READMEを参照する。

## エラー時

検査不合格の候補は公開しない。公開直後の応答障害ではstoreの現行版と要求を照合して結果を確定する。
失敗時はrootの `private/author.log` と `private/gemini/` をローカルで調べ、
briefや入力素材に原因があれば修正して同じrootで明示的に `create` を再実行する。モデルやeffortは変えない。
既に初版があれば `document_exists` になるので `open` を使う。未完了stageを公開用にコピーしない。

閲覧中の失敗では現行版を読み続け、エラーに応じて入力や対象を直してUIの再試行を使う。
新しいrequest IDと現在のbaseを使い、同じ要求の二重送信や自動再送をしない。
`image_generation_unavailable` は初版限定画像生成の方針による未完了であり、再送で生成可能にはならない。
利用者が別の要求として望む場合に、既存画像の解説やSVG/グラフの更新を行う。
停止・再起動後のinterruptedも成功と扱わず、必要なら明示的に再試行する。
