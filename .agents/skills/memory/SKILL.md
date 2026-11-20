---
name: memory
description: Read, write, compress and delete the shared agent memory in .agents/memory/ (SUMMARY.md first, then layer indexes). Use at the start of every task, and whenever the human user gives an instruction or judgment that should not have to be repeated.
---

# 共有memory

Claude・Codexなど、このrepoで働くすべてのエージェントが読み書きする、git管理のmemory。目的は、同じ落とし穴を踏まないこと、そして**ユーザーに同じ指示を繰り返させないこと**である。各ツールのローカルmemory（Claude Code の auto-memory など）には、ユーザー個人の環境（パス、アカウント、マシン固有の設定）だけを置く。

この手順自体も仮説である。運用して合わなければ、このファイルを直す。

## memoryは規則ではない

- memoryは**検証待ちの観察**である。コードやREADMEと食い違ったら、コードとREADMEが正しい。食い違いを見つけたら、そのエントリを直すか削除する。
- 古いもの、反証されたもの、固定観念になって判断を縛っているものは、ためらわず削除する。git履歴が保険になる。
- 繰り返し効いた観察は、規則（`AGENTS.md`）、手順（skill）、仕様（README）へ昇格させ、memoryからは削除する。

## 読む

1. 作業を始める前に、必ず [SUMMARY.md](../../memory/SUMMARY.md) を読む。全体の要点はここにある。
2. 作業に関係しそうな層の `index.md` を開き、関係するエントリだけを読む。全件は読まない。
3. エントリの内容に頼る前に、`evidence` の方法で今も成り立つかを確かめる。確かめたら `last_verified` を更新してよい。

公開する内容は、[専用CLI](../../ops/shared_memory/README.md) が取得する最新の `main` に対して確かめる。

## 層

ディレクトリが種類を表す。1事実1ファイルで、各層の `index.md` に1行ずつ載せる。

| 層 | 書くもの |
|---|---|
| `workflow/` | 作業の進め方のコツ（手順書になるほど大きくないもの） |
| `decision/` | ユーザーの一般的な判断・好み（進め方、聞き方、何を残すか、PRの見方） |
| `environment/` | このマシン・OS・ハードウェアの癖 |
| `gotcha/` | 気づきにくい落とし穴、静かに失敗する挙動 |

`decision/` の記録対象と出典の確認は、[ユーザーの指示を記録する](#ユーザーの指示を記録する)に従う。

## 書く

次の条件をすべて満たすものだけを書く。置き場所の地図は [.agents/README.md](../../README.md#ファイル地図情報の住み分け) にある。

- 再利用できる（次のセッションや別のエージェントが、同じ落とし穴を踏む・同じ指示を受ける）
- コード・README・skill・knowledge を読んでも分からない
- 作業の進捗や状態ではない（進捗は issue / PR に書く）
- 個人情報、トークン、メールアドレス、私的なURLやID、マシン固有のホームパスを含まない（公開repo）

### ユーザーの指示を記録する

このskillで「ユーザー」と呼ぶのは**人間本人**である。「ユーザーの指示」として記録する対象は、人間本人が会話・issue・PRで出した指示と判断に限る。親エージェントの委任・修正指示、reviewer agentの指摘、ほかのエージェントの発言は対象外である。親が指摘を妥当と判断したことも、人間の判断に読み替えない。

記録する前に、発言者と元の発言を確認する。

- 会話では、人間本人との直接のやり取りを確認する。子へ渡された親の指示文は、会話の `user` ロールで届いても人間の発言とは扱わない。
- issue・PRでは、[人間とagentが同じGitHub loginを使う制約](../parent-session/references/finish.md)を踏まえ、loginだけで発言者を判定しない。元の本文・コメントと前後の文脈、人間の確認発言を引用した合意記録を読み、指示・判断を出したのが人間本人だと確かめる。agentの「ユーザーが決めた」という要約だけでは確認済みにしない。
- 委任された子が、親の指示文に引用された人間の発言を記録したい場合は、人間の元の発言へたどれるときだけ書く。たとえば、[parent-sessionの要件合意](../parent-session/references/ownership.md)で人間の会話を引用した親issueの合意コメントを開き、引用内容と記録する判断を照合する。
- `source` には、確認した人間の元の発言を特定できる出典を書く。issue・PRなら該当する本文・コメントのURL、直接の会話なら日付と指示の文脈を記す。会話の引用を記録した合意コメントを使った場合は、そのコメントのURLを記し、親の委任・修正指示やreviewerの指摘を代わりの出典にしない。

人間本人の発言だと確かめられない場合は、`decision/` へ人間の判断として追加・更新しない。agentの指摘や対応の経緯はissue・PRに残す。

人間本人の指示・判断だと確認できたら（人間本人がPRのレビューで「こう直して」と指示したときなど）、その場で再利用性を考える。

1. この指示は、今回の差分だけでなく、今後の別の作業・別のPRにも当てはまるか。当てはまらなければ書かない。
2. 当てはまるなら、今回の具体例から一般化した判断として書く。発言をそのまま写さず、何を望み、何を嫌うかを抽象化する。ただし、今回だけの指定（置き場所の個別指定など）を全体の規則に広げない。
3. 既存エントリと重なるなら、新しく作らず、そのエントリを書き換えて `last_verified` を更新する。同じ指示が繰り返されたことは、そのエントリの重要度が高い証拠なので、本文で明確にする。
4. 複数の作業に使える人間の判断を `decision/` に置く。

共有4層の追加・更新・削除はすべて [専用CLI](../../ops/shared_memory/README.md) で公開する。上の出典確認と「書く」の条件を満たしたら、作業中に得た観察も同じ経路で独立して反映する。CLI が機械検査を行い、共有memory専用のPRを作成してマージする。

### 形式

ファイル名は `<kebab-case>.md`。CLI は JSON の spec から `title`、`summary`、`applies_to`、`source`、`created`、`last_verified`、`evidence`、`citations` の frontmatter と Markdown 本文を生成する。`summary` は `SUMMARY.md` に載る1行の要点。`applies_to` は `all`、`claude`、`codex`、`gemini` のいずれかを選ぶ。

`citations` には観察の根拠になったコード・README・規則の相対パスを指定する。該当する参照が無ければ `[]` を使う。形式の詳細と実行例は [CLI README](../../ops/shared_memory/README.md#コマンド)、サイズと行数の上限は [schema.py](../../ops/shared_memory/schema.py) を参照する。[SUMMARY.md](../../memory/SUMMARY.md) と各層の `index.md` は公開時に自動生成する。

## 利用・変更時の自己修復（JIT）

1. エントリを使うときは `evidence` と `citations` を開いて、観察と根拠を照合する。コード・README・規則を変更するときは、対象パスを `rg` で共有4層のエントリから探し、アンカーや記述への影響を確認する。
2. 矛盾や参照切れを見つけたら、今も成り立つ観察は `shared_memory update` で本文・根拠・引用を修正し、確認した日を `last_verified` に記録する。反証、廃止、仕様への昇格で役目を終えた観察は `shared_memory delete` で削除する。どちらも理由と、読んだ版の生バイトの SHA-256 を `expected_sha256` に渡す。対象が変わっていた場合は最新版から判断し直す。[CLIの形式](../../ops/shared_memory/README.md#コマンド)
3. 新しい引用先は公開時点の `main` に存在するものを選ぶ。まだマージされていない変更で引用先が消える場合は、先に `main` にある有効な引用先へ差し替えるか、不要になった観察を削除する。必要に応じて `--dry-run` で照合する。実際の公開は [専用CLI](../../ops/shared_memory/README.md#コマンド) から行う。

## 圧縮

memoryは毎日読まれるので、大きくなるほど高くつく。`SUMMARY.md`、各 `index.md`、各エントリの上限は [共有memoryのスキーマ](../../ops/shared_memory/schema.py) にある。上限に近づいたら、次の順で圧縮する。

1. 期限切れ・反証済み・コードやREADMEで分かるようになったエントリを削除する。
2. 同じ事柄を言っている複数のエントリを1つに統合する。
3. 繰り返し効いているエントリを、規則・skill・READMEへ昇格させて削除する。
4. 関係する更新と削除を `shared_memory batch` でまとめて公開する。

## 定期保守（AutoDream）

期限・壊れた引用・上限の使用量は `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory audit` で確認する。検証期限の定義は [schema.py](../../ops/shared_memory/schema.py) を参照する。人間の判断は、その後の指示と矛盾しないかを確かめる。

AutoDream は経過日数と `main`・共有memoryの更新件数を組み合わせたトリガーや、手動実行、点検で見つかった問題を契機として、全4層を同じ条件で整理する。Orient → Gather Signal → Consolidate → Prune & Index の順で状況と根拠を確認し、重複や矛盾を整理した操作を専用CLI経由で反映する（#2326で実装中）。生成文書だけがずれていた場合は `shared_memory repair-index --reason <理由>` で公開する。

## 運用レポート専用memory

[ops-report/SUMMARY.md](../../memory/ops-report/SUMMARY.md) → `index.md` → 観察エントリは、
全体調査だけが読む独立領域。この領域の読み取り・保存の実装は
[ops-report README](../../ops/ops-report/README.md#専用memoryの保存) を正本とする。

- 注目領域、根拠付き観察、未解決仮説、明示された見送り理由、次回の再検証条件、関連レポート/提案/issueへの参照を残す。
  issueの進捗や提案本文はリンクで参照し、複製しない。観察は規則ではなく、次回に確かめ直す仮説である。
- 1観察1ファイル。専用のfrontmatterは `title`、`applies_to: "all"`、`source`、`created`、`last_verified`、`evidence` を持つ。
  親スクリプトが作成日・完了公開レポートの参照・索引を生成する。子の調査担当は読み取りだけで、更新内容は親へJSONで返す。
  部分公開中は更新を保存せず、未完了の範囲を検証済みの観察にしない（再開・保存の条件は上記README）。
- 毎回、`evidence` と再検証条件を確認する。再検証条件を満たした観察と、最終確認から90日を過ぎた観察を優先する。
  確かめたものだけ根拠と `last_verified` を更新し、未検証なら日付を進めず、仮説のまま残すか削除する。
- 大きさの上限は [専用領域の文書テスト](../../../tests/e2e/agents_ops/test_agents_docs.py) にある。上限に近づいたら、反証・期限切れを削除し、
  重複を統合し、仕様や手順へ昇格した観察を削除する。削除はエントリと索引を同時に変え、JSONの `delete[].reason` に理由を書く。
  残った観察に合わせて専用 `SUMMARY.md` を書き直す。

例: CIの待ち時間が長いという観察には、計測したrunへの参照・未解決の原因仮説・人間が対策を見送った理由への参照・
「CI構成が変わったら再計測」の条件を残す。再計測で解消を確認したら、確認日を進めて固定観念にせず削除する。
