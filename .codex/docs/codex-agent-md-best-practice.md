# Codex AGENTS.md ベストプラクティス

## 公式例

OpenAI 公式の `AGENTS.md` ガイドは、まず **グローバルな個人ルールを `~/.codex/AGENTS.md` に置き、リポジトリ固有のルールを repo 直下の `AGENTS.md` に置き、さらに必要ならサブディレクトリに `AGENTS.override.md` を置く** という構成を示しています。Codex は起動時にそれらを **global → repo root → deeper directory** の順に読み込み、現在の作業ディレクトリに近いものほど後勝ちで効きます。空ファイルは無視され、合計サイズは既定で 32 KiB までです。 ([OpenAI Developers][1])

```md
# ~/.codex/AGENTS.md

## Working agreements

- Always run `npm test` after modifying JavaScript files.
- Prefer `pnpm` when installing dependencies.
- Ask for confirmation before adding new production dependencies.
```

```md
# AGENTS.md

## Repository expectations

- Run `npm run lint` before opening a pull request.
- Document public utilities in `docs/` when you change behavior.
```

```md
# services/payments/AGENTS.override.md

## Payments service rules

- Use `make test-payments` instead of `npm test`.
- Never rotate API keys without notifying the security channel.
```

この例がよいのは、**個人の恒常ルール**、**repo 全体の共有ルール**、**局所ディレクトリの例外ルール** がきれいに分かれているからです。Codex は `AGENTS.md` を作業前に自動で読み込むため、毎回同じ前提を prompt に書き直さなくて済みます。また、より深いディレクトリの指示が強く効くので、特定パッケージだけ厳しめのルールを持たせる設計に向いています。 ([OpenAI Developers][1])

---

## ベストプラクティス

### 1. AGENTS.md は「毎回守ってほしい永続ルール」だけを書く

OpenAI は `AGENTS.md` を、repo に持ち歩ける **durable project guidance** と位置づけています。向いているのは、ビルド・テスト・lint コマンド、レビュー期待値、repo 固有の慣習、禁止事項、done の定義のように、**毎回の作業で繰り返し必要になる規則** です。 ([OpenAI Developers][2])

これがよい理由は、`AGENTS.md` が長期的な作業前提を担当し、毎回の prompt はその回だけの目標や背景に集中できるからです。OpenAI も、うまくいった prompting pattern は prompt に閉じ込めず `AGENTS.md` に移すことを勧めています。 ([OpenAI Developers][3])

### 2. 短く、正確に、実務的に書く

OpenAI は **“Keep it small”**、**“A short, accurate AGENTS.md is more useful than a long file full of vague rules.”** という方針を示しています。長文化したら task-specific な別 markdown に逃がし、`AGENTS.md` 本体は簡潔に保つのが推奨です。 ([OpenAI Developers][2])

これがよい理由は、Codex が `AGENTS.md` を会話の上流で読み込むため、曖昧で冗長な文書は routing と実行の両方を鈍らせるからです。さらに、読み込み上限は既定で 32 KiB なので、肥大化は実際に不利です。 ([OpenAI Developers][1])

### 3. グローバル・repo・サブディレクトリで責務を分ける

OpenAI の discovery 仕様では、`~/.codex/AGENTS.md` が個人ルール、repo root の `AGENTS.md` が共有ルール、深いディレクトリの `AGENTS.md` または `AGENTS.override.md` が局所ルールを担当します。深い階層ほど後に読み込まれ、実質的に優先されます。 ([OpenAI Developers][1])

これがよい理由は、ルールの適用範囲を自然に分けられるからです。たとえば「全体では `pytest`、ただし `services/payments` だけは `make test-payments`」のような構造を、無理なく表現できます。GitHub integration でも、Codex は変更ファイルに最も近い `AGENTS.md` を重視すると説明されています。 ([OpenAI Developers][1])

### 4. `AGENTS.override.md` は「その階層で上書きしたいときだけ」使う

同じディレクトリに `AGENTS.override.md` がある場合、その階層では `AGENTS.md` より override が優先されます。OpenAI の例でも、`services/payments/AGENTS.md` は override があるため無視されます。 ([OpenAI Developers][1])

これがよい理由は、例外ルールを明示的に扱えるからです。通常の追加ルールなら `AGENTS.md` を使い、**そのディレクトリでは通常ルールを差し替えたい** ときだけ override を使う、という使い分けが分かりやすいです。 ([OpenAI Developers][1])

### 5. 内容は「どう動くべきか」が一目で分かる項目にする

OpenAI の best practices では、良い `AGENTS.md` は次を含むとされています。**repo layout、実行方法、build/test/lint、engineering conventions と PR expectations、constraints / do-not rules、done の定義と検証方法** です。 ([OpenAI Developers][3])

これがよい理由は、Codex が「何を変えるか」だけでなく「どこを見て」「どう検証して」「どこまでやれば完了か」を理解しやすくなるからです。結果として、余計な推測や無関係な探索が減ります。 ([OpenAI Developers][3])

### 6. 同じミスが繰り返されたら AGENTS.md を更新する

OpenAI は、agent が同じ誤りを繰り返す、毎回同じレビュー指摘が出る、読むファイルが多すぎる、といった摩擦が出たら **AGENTS.md にルールを追加して永続化する** ことを勧めています。これは feedback loop と明示されています。 ([OpenAI Developers][2])

これがよい理由は、修正をその場限りの会話に閉じ込めず、次回以降の実行品質に変えられるからです。OpenAI は、GitHub 上でも `@codex add this to AGENTS.md` のように更新を依頼できると案内しています。 ([OpenAI Developers][2])

### 7. 反復ワークフローは AGENTS.md ではなく Skills に寄せる

OpenAI の customization 概念では、**AGENTS.md は行動の方針**、**Skills は再利用可能な workflow**、**MCP は外部システム接続**、**Subagents は役割分担** です。これらは競合ではなく補完関係です。 ([OpenAI Developers][2])

これがよい理由は、役割を混ぜない方が保守しやすいからです。たとえば「必ず lint を走らせる」は `AGENTS.md`、`gh` で issue を作る定型手順は Skill、という分け方が自然です。 ([OpenAI Developers][2])

### 8. まず `/init` を使い、その後で実態に合わせて編集する

OpenAI は CLI の `/init` を、現在のディレクトリに starter `AGENTS.md` を作る quick-start としています。ただし、そのまま使うのではなく、実際の build / test / review / ship の流れに合わせて編集することを勧めています。 ([OpenAI Developers][3])

これがよい理由は、白紙から始めるより速く、しかも運用に合わせて現実的に育てやすいからです。初期化は入口、運用で磨くのが本筋です。 ([OpenAI Developers][3])

### 9. 読み込み確認とトラブルシュートの手順を持つ

OpenAI は、`codex --ask-for-approval never "Summarize the current instructions."` や `codex --cd subdir --ask-for-approval never "Show which instruction files are active."` のように、実際にどの instruction file が効いているか確認する方法を示しています。空ファイルは無視され、想定外の guidance が出るときは上位の `AGENTS.override.md` や `CODEX_HOME` を疑うのが基本です。 ([OpenAI Developers][1])

これがよい理由は、`AGENTS.md` は自動読込なので、効いていない原因が「ファイル名」「配置」「override」「別 home directory」に分かれやすいからです。確認コマンドを持っておくと、設定ミスとモデルの問題を切り分けやすくなります。 ([OpenAI Developers][1])

---

## 推奨テンプレート

```md
# AGENTS.md

## Repository layout
- `src/` が本体コード
- `tests/` が自動テスト
- `docs/` が公開ドキュメント

## Working rules
- 変更前に関連ファイルを読み、既存パターンに合わせる
- 新規依存追加は確認を取る
- 破壊的変更は避ける。必要なら明示する

## Build / test / lint
- Build: `make build`
- Test: `pytest -q`
- Lint: `ruff check .`

## Review expectations
- 変更理由を短く説明する
- 影響範囲を明示する
- 必要ならテストを追加する

## Done when
- 関連テストが通る
- lint が通る
- 動作変更があれば docs を更新する
```

この形がよいのは、OpenAI が推奨する主要項目を短くカバーできるからです。特に **repo layout / commands / conventions / done** がそろうと、Codex の探索、実装、検証、報告が安定しやすくなります。 ([OpenAI Developers][3])

---

## 避けるべき書き方

* 長すぎて task-specific detail まで全部書く
* 抽象的で検証不能な指示を書く
* repo 全体のルールと局所例外を 1 ファイルに混ぜる
* workflow そのものまで全部 `AGENTS.md` に書く
* 同じ失敗が出ても更新しない

これらは、OpenAI が勧める **短く実務的な永続ガイダンス** という設計から外れます。大きくなりすぎたら別 markdown や Skills に分けるのがよいです。 ([OpenAI Developers][3])

---

[1]: https://developers.openai.com/codex/guides/agents-md/ "Custom instructions with AGENTS.md – Codex | OpenAI Developers"
[2]: https://developers.openai.com/codex/concepts/customization/ "Customization – Codex | OpenAI Developers"
[3]: https://developers.openai.com/codex/learn/best-practices/ "Best practices – Codex | OpenAI Developers"
