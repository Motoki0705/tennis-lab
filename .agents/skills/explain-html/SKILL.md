---
name: explain-html
description: Build an explanation HTML page with Gemini and open a local reader where questions and instructions update the page itself. Use for explanations that benefit from diagrams, numerical graphs, or progressive detail, including reopening and revising an existing explanation.
---

# 説明をHTMLで育てる

Codexは根拠・論理・図表と検証を担当し、本文・見出し・図の解説は指定Geminiで執筆する。
質問への回答もHTML本文へ統合する。汎用サイト作成やホスティングはこの手順の範囲外。

最初に [runtime README](../../ops/explain_html/README.ja.md) の「Geminiの執筆と編集」、
「図と画像生成」を読む。wire型・モデル/effort・検証・版保存・公開境界の正本は
runtimeであり、このskillでschemaやpinを複製しない。CLIとbriefの準備は
[初版と運用](references/authoring.md) を使う。

## 読み手と説明経路を決める

依頼と既存資料から、読み手の前提知識、理解・判断してほしいこと、説明対象を把握する。
結果を左右する情報だけが不足している場合に質問し、分かる部分の根拠整理は進める。
既存文書の修正なら、そのroot・document IDを探し、再生成せず開く。

根拠には出典、数値と単位、条件、未確認事項を添える。実測・仮想試算・導出・概念画像を区別し、
文体変更で失ってはいけない事実をWritingContext.evidenceへ渡す。非公開情報は公開Documentに入れない。

**A「段階的に深掘りする」**を理解の主経路にする。目的と全体像から、理由、具体例、必要な詳細へ
進める。主張を支える理由は本文に残し、任意の長い導出や背景をdetailsへ置く。用語は初出で紹介し、
図の近くで「どこを見るか」「何が分かるか」を説明する。短さのために必要な文脈を削らない。

以下は必要なときに組み合わせる。全部を採用する必要はない。

| 経路 | 採用すると理解が進む場面 | 設計の要点 |
|---|---|---|
| B: 疑問を追う | 読み手がつまずく前提や反例がある | 疑問を理由・例へつなげ、回答を本文へ蓄積する |
| C: 操作して確かめる | パラメータの変化が説明に効く | 予想・操作・観察・理由をつなぎ、初期表示だけでも読めるようにする |
| D: 比較して判断する | 条件によって選択が変わる | 同じ比較軸と条件、トレードオフ、追加調査事項を示す |

章数・配色・カード・カラム・文字数は固定しない。題材と読み順からレイアウトを選ぶ。
既存のページを更新するときは、局所的な質問なら周辺へ追記し、全体の組み直しはその指示があるときに行う。

## 表現を準備し、Geminiへ渡す

- **生成画像**: 場面や比喩の直感を助ける場合に使う。初版を作るCodex対話中にimagegen skillを読み、
  利用可能な組込みimagegenで生成する。出力を文書のassetsへコピーし、実際の画像を確認して
  `images.import_image` でimportする。caption・alt・generated_conceptの来歴を付ける。
- **SVG**: 厳密な構造・幾何関係には決定的に描いたSVGを `visuals.import_svg` でimportする。
  生成イラストを数値や幾何の根拠にしない。
- **グラフ**: 数値比較や操作はruntimeが対応する宣言的なグラフを用いる。元データ、軸、単位、範囲、
  前提を揃える。未対応のグラフは任意JSを埋めず、静的な図か対応追加が必要なことを説明する。

briefには説明目的・読者・instruction、根拠、準備したassets/figures、文章とレイアウトの意図を渡す。
構文の制約はruntime READMEを参照する。Geminiの本文をCodexの代筆へ切り替えない。
helperはadapterの指定設定で執筆と独立した意味検査を実行し、同じ最終候補をstageする。
さらにvisual検査とChromiumでの描画検査を行い、全検査を通った初版だけをimportする。

## 開いて更新する

`scripts/explain_html_author.py create` で初版を作り、`open` でserviceと固定reader UIを開く。
CLIのURLを利用者へ渡す。本文の見出し・図・選択範囲、またはページ全体を対象に質問や変更を送れる。
処理中も現行版を読み続け、更新後の変更案内から必要な場所へ移動する。履歴で旧版を読み、undoで
旧版を新しい版として復元できる。起動URLのtokenをHTML・画像・exportや共有資料へ転記しない。

**閲覧中の新規AI画像生成は利用できない。** serviceは `image_generation_unavailable` として
理由付きの未完了にし、要求全体を公開せず現行版を保持する。画像生成の成功として文章/SVGを代用しない。
既存画像の再配置・解説修正、SVG・対応グラフの更新は可能。初版の対話imagegenが使えることを
serviceからの生成能力と読み替えない。別サービスやモデルへ無断で切り替えない。

## 読者の立場で確かめる

実ブラウザで、全体像・理由・具体例・図の読み方・詳細への到達を確かめる。数値、単位、出典、条件を
元の根拠へ照合し、意味検査のpassedだけで説明の十分さを断定しない。狭幅、keyboard、alt、contrast、
reduced-motionを確認する。自動描画検査は表示の一部の不具合を検出するもので、読後評価の代替ではない。

質問の回答が本文に現れること、全体再構成でも根拠を失わないこと、undo・再open・停止後のexportを
必要な範囲で試す。静的版が閲覧専用であることを本文の小さな案内で分かるようにし、書き出した一式を
service停止後に開く。失敗時の再試行は [初版と運用](references/authoring.md) に従い、現行版を保持する。

実機確認では生成と意味検査の両方の呼出数、時間、取得できるusage、実効model/effortの証拠と
成果物のパスを残す。改善の予算・終了条件を決め、無制限に再生成しない。人間の読後評価は人間に記入して
もらい、代理で評価済みにしない。認知負荷を客観測定したという主張に置き換えない。
