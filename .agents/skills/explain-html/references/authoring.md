# 初版と運用

この文書はskillのhelperとbriefの準備を説明する。文書・根拠・assets/figuresの契約、許可されるHTML/CSS、
グラフ定義、model/effort、HTTP・保存の仕様は [runtime README](../../../ops/explain_html/README.ja.md) を参照する。

## 準備

repo rootで `.venv/bin/python` を使う。Chromiumによる実描画検査にはrepoの既存Playwright依存が必要。
未導入なら既存lockで `npm ci --prefix scripts/ci_metrics` と
`scripts/ci_metrics/node_modules/.bin/playwright install chromium --only-shell` を実行する。
必要に応じて `PLAYWRIGHT_MODULE` と `CHROMIUM_PATH` で導入済みの実体を明示できる。
helperは利用不可のブラウザ検査をpassedにしない。

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
管理部品・figure修復・数式・狭幅の描画はruntimeのwriter/compile契約に従う。
helperは準備されたグラフがある場合、現行managed rendererのmetadataと実bytesを追加する。
初版用に自分で用意したrenderer/JSへ差し替える入口はない。

## 生成・起動・再open・停止・export

```sh
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py create --root outputs/explain-html/example/runtime --brief outputs/explain-html/example/brief.json
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py publish --root outputs/explain-html/example/runtime --stage STAGE_ID --review-file outputs/explain-html/example/codex-review.json
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py open --root outputs/explain-html/example/runtime --document example --browser
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py status --root outputs/explain-html/example/runtime
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py stop --root outputs/explain-html/example/runtime
.venv/bin/python .agents/skills/explain-html/scripts/explain_html_author.py export --root outputs/explain-html/example/runtime --document example
```

`create` は既存documentを上書きしない。Geminiの執筆/独立意味検査→非公開stage→実bytesのvisual/ブラウザ
検査を行い、`state=awaiting_codex_review` とstage ID・candidate digest・review directoryを返す。
この時点で公開初版はなく、`open` はできない。directory内の `candidate.json` と
`content/index.html`・assetsを読み、[Codexの根拠照合](claim-review.md) を実行する。
`turns.json` は成功したwriter/既存figure修復/verifierの設定・時間・usageの記録で、rawログはruntimeのprivateに残る。

実際の照合を終えてから、helperの `CodexReview` 型で確認記録をJSONへ保存する。型の正本はhelperにあり、
`reviewer="codex"`、candidate digest、全Evidence IDそれぞれの確認理由、非同期・固定量/感度・条件付きの
一般化の確認記録、総合CheckResultを持つ。題材にない観点も、その理由を記す。
確認記録は例えば上の `outputs/explain-html/example/codex-review.json` へ置き、helperが保存する
review directory内の `codex-review.json` を編集しない。schemaは同じ型の `model_json_schema()` で参照できる。

`publish` は確認記録のdigest・Evidence IDを同じstageへ照合し、合格かつissueなしの場合だけ初版をcommitする。
不合格の記録はprivateに保持し、同じ候補の判定を書き換えて公開することはできない。
Codexの考察を機械的に証明するものではなく、信頼するローカル実行者による確認の記録である。
`open` は公開済み文書だけを開き、生成を再実行しない。
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
Codexが誤りを見つけた場合は不合格の確認記録を `publish` へ渡して保存し、誤った引用と修正理由を
writerへのinstructionへ追加する。新しい候補を作り直して全検査を行い、新しい確認記録で公開する。
既に初版があれば `document_exists` になるので `open` を使う。未完了stageを公開用にコピーしない。

閲覧中の失敗では現行版を読み続け、エラーに応じて入力や対象を直してUIの再試行を使う。
新しいrequest IDと現在のbaseを使い、同じ要求の二重送信や自動再送をしない。
`image_generation_unavailable` は初版限定画像生成の方針による未完了であり、再送で生成可能にはならない。
利用者が別の要求として望む場合に、既存画像の解説やSVG/グラフの更新を行う。
停止・再起動後のinterruptedも成功と扱わず、必要なら明示的に再試行する。

agyがnative SUCCESSや候補を出力した後も終了せず、writerの900秒上限でtimeoutになる既知の現象がある
（先行の約30回の実機呼出で2回）。候補の出力だけを正常終了と読み替えず、未採用streamからimportしない。
version-afterを含むadapterの照合・独立意味検査・描画検査が完了していなければ初版を公開しない。
privateログ、終了/未終了と最終出力の時刻、request IDを保存し、残る呼出予算を確認して明示的に再実行する。
上限や検査を緩めず、閲覧中の失敗では現行版を保持する。
