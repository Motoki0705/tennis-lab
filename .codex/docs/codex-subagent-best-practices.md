# Codex Subagent ベストプラクティス

## 公式例

OpenAI 公式は、PR レビューを **3 つの責務に分けた custom agent 構成**を例として示しています。`pr_explorer` はコードパス調査、`reviewer` は correctness / security / tests の確認、`docs_researcher` はドキュメントや API 仕様の確認を担当します。また、`[agents]` では `max_threads = 6`、`max_depth = 1` を使い、各 agent の `sandbox_mode` は用途に応じて設定されています。 ([OpenAI デベロッパー][1])

```toml
# .codex/config.toml
[agents]
max_threads = 6
max_depth = 1
```

```toml
# .codex/agents/pr-explorer.toml
name = "pr_explorer"
description = "Read-only codebase explorer for gathering evidence before changes are proposed."
model_reasoning_effort = "medium"
sandbox_mode = "read-only"
developer_instructions = """
Stay in exploration mode.
Trace the real execution path, cite files and symbols, and avoid proposing fixes unless the parent agent asks for them.
Prefer fast search and targeted file reads over broad scans.
"""
```

```toml
# .codex/agents/reviewer.toml
name = "reviewer"
description = "PR reviewer focused on correctness, security, and missing tests."
model_reasoning_effort = "high"
sandbox_mode = "read-only"
developer_instructions = """
Review code like an owner.
Prioritize correctness, security, behavior regressions, and missing test coverage.
Lead with concrete findings, include reproduction steps when possible, and avoid style-only comments unless they hide a real bug.
"""
```

```toml
# .codex/agents/docs-researcher.toml
name = "docs_researcher"
description = "Documentation specialist that uses the docs MCP server to verify APIs and framework behavior."
model_reasoning_effort = "medium"
sandbox_mode = "read-only"
developer_instructions = """
Use the docs MCP server to confirm APIs, options, and version-specific behavior.
Return concise answers with links or exact references when available.
Do not make code changes.
"""
```

## なぜこの例がよいのか

この例がよいのは、**agent ごとの責務が狭く、出力の種類も明確だから**です。OpenAI は subagent を、探索メモ・ログ・テスト出力のようなノイズを親スレッドから切り離し、親 agent を要件整理と最終判断に集中させるための仕組みとして説明しています。これにより **context pollution** と **context rot** を避けやすくなります。 ([OpenAI デベロッパー][2])

また、この構成は **権限を最小化している**点でも優れています。探索役・レビュー役・ドキュメント確認役はいずれも `read-only` で十分であり、Codex は sandbox mode と approval policy を組み合わせて安全に動かす設計です。さらに、ネットワークはデフォルトでオフです。つまり、必要な agent にだけ必要な権限を与えるのが基本です。 ([OpenAI デベロッパー][3])

---

## ベストプラクティス

### 1. agent は狭い責務で作る

1 agent = 1 役割に寄せます。
例: `codebase_explorer`、`reviewer`、`docs_researcher`、`browser_debugger`。

**理由**
狭い責務の agent は、何を調べて何を返すべきかが明確になります。OpenAI の公式例も、探索・レビュー・仕様確認を分離しています。 ([OpenAI デベロッパー][1])

### 2. explorer 系は read-only を基本にする

調査用 subagent は、まず `sandbox_mode = "read-only"` を基本にします。

**理由**
explorer は証拠収集が役割であり、変更権限は不要です。OpenAI の公式例でも `pr_explorer` と `reviewer` と `docs_researcher` は read-only です。安全性の面でも、sandbox と approvals を強めに保つのが基本です。 ([OpenAI デベロッパー][1])

### 3. developer_instructions は「行動境界」を明記する

`developer_instructions` には、役割だけでなく **やってよいこと / だめなこと** を入れます。
例: 「探索モードを維持する」「ファイルとシンボルを引用する」「親に求められない限り修正案を出さない」。

**理由**
subagent は specialized worker なので、境界が曖昧だと勝手に設計提案や編集に踏み込みやすくなります。公式の `pr_explorer` も、修正提案を抑えて証拠収集に集中するよう明示しています。 ([OpenAI デベロッパー][1])

### 4. 親 agent には「要約」を返し、生ログを返しすぎない

subagent の返答は、原則として **要点 + 根拠** にします。
ログ全文、探索の試行錯誤、無関係な grep 結果を大量に返さないようにします。

**理由**
OpenAI は、subagent の価値を「ノイズを main thread から逃がすこと」に置いています。返すものまでノイズだと効果が薄れます。 ([OpenAI デベロッパー][2])

### 5. `max_depth` は原則 1 のままにする

再帰的 delegation が本当に必要でない限り、`max_depth = 1` を維持します。

**理由**
公式は、深い再帰は token 使用量、レイテンシ、ローカル資源消費、予測しづらさを増やすと説明しています。まずは親 → 子 までで十分です。 ([OpenAI デベロッパー][1])

### 6. project 固有の agent は `.codex/agents/` に置く

custom agent は `~/.codex/agents/` または `.codex/agents/` に置けます。
チームで共有したい agent は repo 配下の `.codex/agents/` に置くのが基本です。

**理由**
project-scoped config は repo に閉じた運用ができ、trusted project では project config が user config より優先されます。チームの標準運用を repo に載せやすくなります。 ([OpenAI デベロッパー][1])

### 7. 共通ルールは AGENTS.md、繰り返し手順は Skills に分ける

* **AGENTS.md**: repo 全体の作法、レビュー方針、実行前後のルール
* **Subagent**: 専門役の分担
* **Skills**: 繰り返し使うワークフローや補助スクリプト

**理由**
OpenAI は、AGENTS.md を事前に読む project guidance、Skills を task-specific workflow bundle、Subagents を delegated specialized work として分けています。役割を混ぜない方が保守しやすいです。 ([OpenAI デベロッパー][4])

### 8. `description` は「いつ使う agent か」が分かる文にする

`description` は名前の言い換えではなく、**起動条件が伝わる説明**にします。
例: `Read-only subagent for surveying src/tasks conventions before implementation starts.`

**理由**
OpenAI は `description` を「Codex がその agent をいつ使うべきかを示す人間向けガイダンス」と位置づけています。短くても、用途が見える文が有効です。 ([OpenAI デベロッパー][1])

---

## 推奨テンプレート

```toml
name = "tasks_codebase_explorer"
description = "Read-only subagent for surveying src/tasks conventions before implementation starts."
sandbox_mode = "read-only"
model_reasoning_effort = "medium"
developer_instructions = """
Stay in exploration mode.
Trace the real execution path. Cite file paths and symbols.
Prefer fast search and targeted reads over broad scans.
Do not make code changes.
Avoid proposing fixes unless the parent agent explicitly asks.
"""
```

この形がよいのは、**役割、権限、出力スタイル、禁止事項** がすべて短く入っているからです。公式の `pr_explorer` の設計思想にもかなり近いです。 ([OpenAI デベロッパー][1])

---

## 避けるべき書き方

* 責務が広すぎる agent
  例: 「調査も実装もレビューも全部やる」

* 権限が広すぎる agent
  例: explorer なのに `workspace-write`

* developer_instructions が抽象的すぎる agent
  例: 「うまく調べてください」

* 生ログをそのまま親に返す agent

これらは、subagent を使う意味である **分離・要約・安全化** を弱めます。 ([OpenAI デベロッパー][2])

---

[1]: https://developers.openai.com/codex/subagents/ "Subagents – Codex | OpenAI Developers"
[2]: https://developers.openai.com/codex/concepts/subagents/ "Subagents – Codex | OpenAI Developers"
[3]: https://developers.openai.com/codex/agent-approvals-security/ "Agent approvals & security – Codex | OpenAI Developers"
[4]: https://developers.openai.com/codex/guides/agents-md/ "Custom instructions with AGENTS.md – Codex | OpenAI Developers"
