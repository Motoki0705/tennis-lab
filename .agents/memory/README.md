# 共有memory

Claude・Codexなど、このrepoで働くすべてのエージェントが読み書きする、git管理のmemory。各ツールのローカルmemory（Claude Code の auto-memory など）とは別物である。ローカルmemoryには個人の嗜好と個人の環境だけを置き、プロジェクトの知見はここに書く。

## memoryは規則ではない

- memoryは**検証待ちの観察**である。規則でも仕様でもない。
- コードやREADMEと食い違ったら、コードとREADMEが正しい。食い違いを見つけたエージェントは、そのエントリを直すか削除する。
- 古くなったもの、反証されたもの、固定観念になって判断を縛っているものは、ためらわず削除してよい。git履歴が保険になる。
- 繰り返し効いた観察は、規則（`AGENTS.md`）、手順（skill）、仕様（README）へ昇格させ、memoryからは削除する。memoryに残し続けない。

## いつ読むか

- 作業を始めるときに [INDEX.md](INDEX.md) を読み、関係しそうなエントリだけを開く。全件は読まない。
- エントリの内容を使う前に、`evidence` の方法で今も成り立つかを確かめる。確かめたら `last_verified` を更新してよい。

## 何を書くか

置き場所の地図は [.agents/README.md](../README.md#ファイル地図情報の住み分け) にある。memoryに書くのは、次の条件を満たすものだけである。

- 再利用できる（次のセッションや別のエージェントが同じ落とし穴を踏む）
- コード・README・skill・knowledge を読んでも分からない
- 作業の進捗や状態ではない（進捗は issue / PR に書く）
- 個人情報、トークン、メールアドレス、私的なURLやIDを含まない（公開repo）

## 形式

1事実1ファイル。ファイル名は `<kebab-case>.md`。下の例のキーはすべて必須。frontmatterはYAMLなので、文字列（`#` やバッククォートで始まるものを含む）は二重引用符で囲む。

```markdown
---
title: "1行で事実を述べる（INDEX.md のリンク文字列と一致させる）"
type: gotcha            # gotcha | environment | decision | workflow
applies_to: all         # all | claude | codex
source: "どこで観察したか（#issue / PR / commit / 日付）"
created: 2026-10-10
last_verified: 2026-10-10
evidence: "今も成り立つかを確かめる方法（コマンド、ファイル、行）"
---

観察（何が起きるか）。

**使い方:** 次にどう行動するか。
```

| type | 意味 |
|---|---|
| `gotcha` | 気づきにくい落とし穴、静かに失敗する挙動 |
| `environment` | このマシン・OS・ハードウェアの癖 |
| `decision` | ユーザーの判断で、まだREADMEや規則に載っていないもの |
| `workflow` | 作業の進め方に関するコツ（手順書になるほど大きくないもの） |

## 追加・更新・削除

- **追加**: 既存エントリと重複しないことを INDEX.md で確かめる。ファイルを作り、INDEX.md に1行追加する。`created` と `last_verified` は観察した日付にする。
- **更新**: 内容を確かめ直したら `last_verified` を更新する。事実が変わったら本文を書き換える（変更履歴は書かない。gitにある）。
- **削除**: ファイルと INDEX.md の行を、同じコミットで消す。理由はコミットメッセージに書く。
- **検証期限**: `last_verified` から90日を過ぎたエントリは、週次の運用レポートで削除候補に挙げる。90日という値も仮説であり、運用を見て改訂する。
- INDEX.md と実ファイルの整合、frontmatterの必須キーは `tests/e2e/agents_ops/test_agents_docs.py` が検証する。

memoryの変更もPRで入れる。memoryだけを変えるPRを人間承認の対象に含めるかどうかは、共有基盤パスの一覧（[review/README.md](../ops/review/README.md)、#1058で実装中）で決める。
