# Codex SKILL.md ベストプラクティス

## 公式例

OpenAI 公式は、**「小さな React デモアプリを毎回同じ形で作る skill」** を例として示しています。内容は、`SKILL.md` の front matter に `name` と `description` を置き、その後に **いつ使うか**、**何を作るか**、**手順**、**definition of done** を書く構成です。さらに公式は、この例を **意図的に狭く・具体的にしている** と説明しています。評価可能な条件がないと、skill の良し悪しを判定しにくいからです。 ([OpenAI デベロッパー][1])

```md
---
name: setup-demo-app
description: React のデモアプリを決まった最小構成で作るときに使う。
---

## いつ使うか
UI の試作や再現用に、すぐ動く最小アプリが欲しいとき。

## 何を作るか
- Vite + React + TypeScript
- Tailwind を設定
- 最小の components 構成
- 余計な UI ライブラリは入れない

## 手順
1. テンプレートから作成
2. 依存を入れる
3. Tailwind を設定
4. 最小 UI を作る

## 完了条件
- 開発サーバが起動する
- 必須ファイルがそろう
- 指定した構成になっている
```

この例がよいのは、**skill が 1 つの仕事に絞られており、発火条件と完成条件が明確だから**です。Codex の skills は、最初に `name` と `description` だけを見て発見・選択し、必要になったときだけ `SKILL.md` 本文や `scripts/` などを読みます。つまり、skill の成否は本文の長さよりも、最初のメタデータと「何をいつどう終えるか」の明確さに強く依存します。 ([OpenAI デベロッパー][2])

## ベストプラクティス

### 1. skill は 1 つの仕事に絞る

1 つの skill に複数の役割を詰め込まず、**1 workflow = 1 skill** に寄せます。OpenAI も「Keep each skill scoped to one job」と案内しています。まず 2〜3 個の具体的なユースケースを定め、入力と出力をはっきりさせるのが推奨です。 ([OpenAI デベロッパー][3])

これがよい理由は、Codex が skill を選びやすくなり、使ったあとも「期待どおりに動いたか」を判定しやすいからです。広すぎる skill は説明文も曖昧になり、暗黙発火が不安定になります。 ([OpenAI デベロッパー][2])

### 2. `description` は説明文ではなく「発火条件」として書く

公式は、`description` を **routing contract** と位置づけています。Codex は skill の存在を把握するとき、まず `name`・`description`・path を見て、使うべきかどうかを判断します。したがって `description` には、**何をする skill か** だけでなく、**いつ使うか / いつ使わないか** を入れるのが重要です。 ([OpenAI デベロッパー][2])

よい `description` は、たとえば「PR 文を書く」ではなく、**“大きめの変更が終わったあとに、PR-ready な要約ブロックを作るときに使う”** のように、タイミングと対象を含みます。これにより implicit invocation の精度が上がります。 ([OpenAI デベロッパー][4])

### 3. 本文には「いつ使うか」「何を作るか」「手順」「完了条件」を書く

公式例は、単なる作業メモではなく、**使用条件・成果物・実施順・definition of done** を含んでいます。 ([OpenAI デベロッパー][1])

これがよい理由は、Codex にとっても人間にとっても、skill の目的と終了判定が同じになるからです。特に `definition of done` を入れると、無駄な試行錯誤を減らしやすくなります。 ([OpenAI デベロッパー][1])

### 4. 最初は instruction-only で始める

公式の skill 作成フローでは、`$skill-creator` が **instruction-only をデフォルト推奨** にしています。Codex の best practices でも、最初から全 edge case を詰め込まず、代表的な 1 タスクで skill 化し、必要に応じて育てる方針が勧められています。 ([OpenAI デベロッパー][2])

これがよい理由は、早い段階で script や資産を増やしすぎると、保守対象が増えるわりに routing の問題が見えにくくなるからです。まずは説明と流れを固め、そのあと deterministic な処理だけを script に逃がすのが安定します。 ([OpenAI デベロッパー][3])

### 5. 繰り返しの機械処理だけを `scripts/` に置く

OpenAI の blog では、良い分担として **解釈・比較・報告は model 側、決定的で反復的な shell 作業は `scripts/` 側** と整理しています。さらに API docs 系の guidance では、skill scripts を **小さな CLI のように設計する** ことが勧められています。標準出力は決定的にし、失敗時は明確な usage / error を返し、必要なら既知のファイルパスに成果物を書きます。 ([OpenAI デベロッパー][4])

これがよい理由は、model が得意な判断と script が得意な反復を分離できるからです。たとえば lint, test, log collect, fixed-order verification のような処理は script 化の相性が良いです。 ([OpenAI デベロッパー][4])

### 6. `references/` と `assets/` は「信頼性を上げるときだけ」足す

Codex skills は `SKILL.md` に加え、`scripts/`、`references/`、`assets/`、`agents/openai.yaml` を持てます。ですが公式は、script や追加資産は **信頼性を上げるときだけ** 入れることを勧めています。 ([OpenAI デベロッパー][2])

これがよい理由は、skill の本体は workflow の定義であって、添付物を増やすこと自体ではないからです。まずは本文だけで十分かを見て、曖昧さや再現性の問題が出た部分だけを補強するのがよいです。 ([OpenAI デベロッパー][3])

### 7. 現在の保存先は `.agents/skills` を優先する

現在の Codex skills ドキュメントは、repo・user・admin・system の各レベルで skill を読み込み、repo と user の代表的な場所として **`.agents/skills`** と **`$HOME/.agents/skills`** を案内しています。best practices も同じ保存先を案内しています。 ([OpenAI デベロッパー][2])

一方で、2026-01 の公式 blog には `.codex/skills/...` や `~/.codex/skills/...` の例も残っています。したがって、**新しく運用を始めるなら現行 docs に合わせて `.agents/skills` を優先する** のが自然です。これは docs の現在の記述を優先した実務上の判断です。 ([OpenAI デベロッパー][2])

### 8. implicit invocation を使うなら、境界を狭くする

Codex は skill を **明示的に呼ぶ** ことも、`description` に一致して **暗黙的に選ぶ** こともできます。さらに `agents/openai.yaml` で `allow_implicit_invocation: false` を設定すれば、暗黙発火を止めて明示呼び出し専用にもできます。 ([OpenAI デベロッパー][2])

これがよい理由は、skill によっては勝手に走ってほしくないものがあるからです。たとえば危険な変更や重い検証、外部依存の強い skill は、明示呼び出しだけにした方が扱いやすいです。 ([OpenAI デベロッパー][2])

### 9. 何度も同じ prompt を使うなら skill 化する

OpenAI は、同じ prompt を何度も使ったり、同じ workflow を何度も修正しているなら **それは skill にすべきサイン** だと述べています。典型例として、log triage、release note drafting、PR review、migration planning、incident summary、debugging flow が挙げられています。 ([OpenAI デベロッパー][3])

これがよい理由は、手順を skill に閉じ込めることで、毎回の prompt の長文化を防ぎ、再現性を上げられるからです。安定した後は automation に乗せる、という流れも公式が推奨しています。 ([OpenAI デベロッパー][3])

## 推奨テンプレート

```md
---
name: task-name
description: 何をする skill かに加えて、いつ使うか / いつ使わないかを明記する。
---

## When to use
この skill を使う条件を書く。

## Inputs
必要な入力、前提、対象範囲を書く。

## Outputs
期待する成果物や返答形式を書く。

## Steps
1. 実施順を短く固定する
2. 判断が必要な点を書く
3. 必要なら確認手順を書く

## Definition of done
- 完了条件を列挙
- 検証方法を列挙
```

この形がよいのは、**routing 用の front matter** と **実行用の本文** がはっきり分かれ、しかも skill の終了判定まで含められるからです。Codex はまず metadata を見て skill を選び、選ばれた後に本文を読むので、この二段構えに合わせた構成にするのが合理的です。 ([OpenAI デベロッパー][2])

## 避けるべき書き方

* `description` が広すぎる
  例: 「開発を助ける skill」

* 1 skill に複数 workflow を入れる
  例: 調査、実装、レビュー、PR 作成を全部まとめる

* 完了条件がない
  例: 「いい感じに整える」

* script 化すべき反復処理を本文に長く書く

* 逆に、まだ安定していない手順を最初から script に押し込む

これらは、routing の不安定化、保守コストの増加、評価不能を招きやすいです。公式の guidance は一貫して、**狭い skill、明確な description、必要最小限の補助資産** を勧めています。 ([OpenAI デベロッパー][2])

[1]: https://developers.openai.com/blog/eval-skills/ "Testing Agent Skills Systematically with Evals | OpenAI Developers"
[2]: https://developers.openai.com/codex/skills/ "Agent Skills – Codex | OpenAI Developers"
[3]: https://developers.openai.com/codex/learn/best-practices/ "Best practices – Codex | OpenAI Developers"
[4]: https://developers.openai.com/blog/skills-agents-sdk/ "Using skills to accelerate OSS maintenance | OpenAI Developers"
