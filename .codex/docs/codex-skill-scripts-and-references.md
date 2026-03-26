# Codex Skills における `references/` と `scripts/` の書き方・作り方

## 公式例

OpenAI 公式の curated skill では、`SKILL.md` から **script は「実行するもの」** として、**reference は「必要時に読むもの」** として明示しています。たとえば `speech` skill は description と workflow の両方で `scripts/text_to_speech.py` を指し、詳細は `references/cli.md` を見るように分けています。`render-deploy` skill でも、本文は短く保ちつつ、詳細手順を `references/direct-creation.md` や `references/codebase-analysis.md` に逃がしています。これは OpenAI が説明する「progressive disclosure」の実例です。 ([OpenAI デベロッパー][1])

---

## 基本原則

Codex の skill は、`SKILL.md` を必須とし、必要に応じて `scripts/` と `references/` を持てます。公式の位置づけは明確で、`scripts/` は **決定的で繰り返し発生する処理**、`references/` は **作業中に必要に応じて読む資料** です。`SKILL.md` は workflow の中核だけを書き、重い詳細は `references/` に分離するのが基本です。 ([OpenAI デベロッパー][1])

---

## `SKILL.md` でどう言及するか

### 1. frontmatter には書きすぎない

`SKILL.md` の frontmatter は `name` と `description` だけにします。OpenAI の skill-creator は、**発火条件は description に書き、本文の “When to use” に頼らない** ことを勧めています。つまり、`scripts/` や `references/` の細かな説明は frontmatter ではなく本文側に置きます。 ([OpenAI デベロッパー][1])

### 2. script は本文で「いつ・どう実行するか」を書く

script は `Workflow` や `How to run` の中で、**ファイル名を明示して** 書くのが公式例に近いです。`speech` skill は「bundled CLI を実行する」と書き、具体的に `scripts/text_to_speech.py` を示しています。API cookbook の runnable example でも、`SKILL.md` に `How to run` を置き、依存導入コマンドと `run.py` の実行例、期待する出力先を書いています。 ([GitHub][2])

### 3. reference は本文で「どの場面で読むか」を書く

reference は単に「参考資料がある」と書くのではなく、**どの判断でどの file を読むか** を `SKILL.md` に書きます。公式の `render-deploy` skill は、各 step ごとに `references/direct-creation.md` や `references/codebase-analysis.md` を指しています。OpenAI の skill-creator も、分離した資料を置くときは **SKILL.md から明示的に参照し、いつ使うかをはっきり書く** よう求めています。 ([GitHub][3])

### 4. SKILL.md は短く、詳細は reference に逃がす

OpenAI の guidance は、`SKILL.md` 本文は essentials に寄せ、長くなりすぎたら reference に分割する、というものです。特に **schemas、API docs、詳細手順、例、ポリシー類** は `references/` に置くのが推奨です。`SKILL.md` には core workflow と分岐判断だけを残します。 ([GitHub][4])

### 5. 大きい reference は検索の入口も書く

公式の skill-creator は、reference が大きい場合、`SKILL.md` に **grep 用の検索語や入口** を書くことを勧めています。要するに、reference を置くだけでは不十分で、「何を探すときにこの file を開くか」まで書くのがよい、ということです。 ([GitHub][4])

---

## 推奨パターン

### script の言及パターン

* description では、必要なら「bundled CLI を使う」程度まで触れる
* 本文では `Workflow` または `How to run` に実行タイミングを書く
* 実行コマンド、主要引数、出力先を書く
* 期待する生成物を `Outputs` に書く

この形がよいのは、Codex が **何を実行すべきか** と **どこに成果物が出るか** を迷いにくいからです。公式 cookbook でも `How to run` と `Outputs` を入れるよう勧めています。 ([OpenAI デベロッパー][5])

### reference の言及パターン

* `Workflow` の各段階で「この判断では `references/foo.md` を読む」と書く
* `Advanced`, `Troubleshooting`, `Variants` などを reference に分離する
* `SKILL.md` に詳細を重複させない

この形がよいのは、`SKILL.md` を短く保ちながら、必要時だけ reference を読む progressive disclosure に合うからです。 ([OpenAI デベロッパー][1])

---

## script はどう作るべきか

### 1. 作るべき場面

OpenAI は、`scripts/` を **同じコードを何度も書き直しているとき**、または **決定的な信頼性が必要なとき** に入れるとしています。たとえば固定順での検証コマンド実行、ログ収集、前処理、ファイル変換のような処理です。 ([GitHub][4])

### 2. 設計方針

OpenAI の cookbook は、skill script を **tiny CLI** のように設計することを勧めています。つまり、コマンドラインから実行でき、stdout は決定的で、usage/error をはっきり出し、必要なら既知のパスに成果物を書きます。 ([OpenAI デベロッパー][5])

### 3. SKILL.md との分担

解釈、比較、報告、例外判断は model 側に残し、**機械的な反復処理だけ script に落とす** のが OpenAI の推奨です。つまり、script は「考える」ためではなく「確実に実行する」ために作ります。 ([OpenAI デベロッパー][6])

### 4. 作成手順

OpenAI の skill-creator は、`init_skill.py` で skill の雛形を作る方法を示しており、`--resources scripts,references` のように resource directory を生成できます。追加した script は、**実際に動かしてテストする** ことも明言されています。 ([GitHub][4])

### 5. script に向いているもの

* ファイル変換
* 固定順の build / test / lint 実行
* ログ収集や再実行ファイル生成
* CSV / JSON / PDF の定型処理
* 既知フォーマットのレポート生成

これらは、公式が挙げる「deterministic, repeated shell work」に合います。 ([OpenAI デベロッパー][6])

---

## reference はどう作るべきか

### 1. 作るべき場面

OpenAI は、`references/` を **作業中に参照すべき資料** の置き場としています。具体例は、API 仕様、DB schema、社内ポリシー、詳細 workflow、domain knowledge です。 ([GitHub][4])

### 2. 設計方針

reference は **読ませるための file** です。したがって、実行可能 code ではなく、判断材料としての情報を置きます。OpenAI は、reference の役割を「Codex の process と thinking を支えるための documentation」と説明しています。 ([GitHub][4])

### 3. SKILL.md との分担

OpenAI の推奨は明快で、**情報は SKILL.md と reference に重複して置かない** ことです。SKILL.md には essential procedural guidance だけを置き、詳しい schema や例や派生ケースは reference に逃がします。 ([GitHub][4])

### 4. reference に向いているもの

* API endpoint の一覧
* DB / table schema
* リポジトリ固有の build / deploy ルール
* 会社固有の style / policy / NDA
* 長い troubleshooting guide
* 分岐ごとの詳細手順

これは公式の examples と一致しています。 ([GitHub][4])

### 5. 避けるべきもの

OpenAI の skill-creator は、skill には **README.md、INSTALLATION_GUIDE.md、QUICK_REFERENCE.md、CHANGELOG.md** のような余計な補助文書を増やさないよう勧めています。reference は汎用 docs 置き場ではなく、**その skill の実行に直接必要な資料** に絞るべきです。 ([GitHub][4])

---

## 推奨テンプレート

````md
---
name: my-skill
description: Use when Codex needs to run a repeatable workflow for X. Use this skill for A and B. Do not use it for C.
---

# My Skill

## Workflow
1. Confirm the target and inputs.
2. If you need the exact command/flags, run `scripts/do_x.py`.
3. If you need the detailed schema or edge cases, read `references/schema.md`.
4. Write outputs to `output/...`.
5. Validate the result.

## How to run
```bash
python -m pip install -r requirements.txt
python scripts/do_x.py --input <path> --outdir output
````

## Outputs

* `output/result.json`
* `output/report.md`

## Reference guide

* Read `references/schema.md` when mapping fields.
* Read `references/troubleshooting.md` if validation fails.

```

この形がよいのは、script を **実行入口** として、reference を **判断資料** として分けて書けるからです。OpenAI の `speech`、`render-deploy`、`csv_insights_skill` の書き方と整合します。 :contentReference[oaicite:19]{index=19}

---

[1]: https://developers.openai.com/codex/skills/ "Agent Skills – Codex | OpenAI Developers"
[2]: https://github.com/openai/skills/blob/main/skills/.curated/speech/SKILL.md "skills/skills/.curated/speech/SKILL.md at main · openai/skills · GitHub"
[3]: https://github.com/openai/skills/blob/main/skills/.curated/render-deploy/SKILL.md "skills/skills/.curated/render-deploy/SKILL.md at main · openai/skills · GitHub"
[4]: https://github.com/openai/skills/blob/main/skills/.system/skill-creator/SKILL.md "skills/skills/.system/skill-creator/SKILL.md at main · openai/skills · GitHub"
[5]: https://developers.openai.com/cookbook/examples/skills_in_api/ "Skills in OpenAI API"
[6]: https://developers.openai.com/blog/skills-agents-sdk/ "Using skills to accelerate OSS maintenance | OpenAI Developers"
