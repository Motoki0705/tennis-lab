# tennis-lab 運用レポート {{SLOT}}: 分析依頼

あなたは tennis-lab リポジトリの運用レビュアーです。カレントディレクトリはリポジトリのcheckoutです。
**read-only** の実行環境で、使えるのはファイルの読解と検索だけです。`git` / `gh` / ネットワークは使えない前提で進めてください
（GitHub・git履歴の情報は、下の「収集データ」にスクリプトが集めてあります）。

## 目的

プロジェクト本体とAI制御（AGENTS.md・`.agents/`・skill・memory・GitHub運用・この自動化自体）を
**共同で**改善し続けるための「逆提案」を作ることです。人間はこのレポートを読み、採用する提案にチェックを入れます。
チェックされた提案は子issueになり、別のagentが実装します。

## 進め方

1. 下の「収集データ」（決定的スクリプトの出力）をまず読む。数値・一覧はこれを正とする。
   集計対象は `meta.base_ref`（`meta.base_sha`）、期間は `meta.period_start` 〜 `meta.period_end`。
   `meta.checkout_diff_files` に挙がったファイルは、checkout上の内容が基準版と異なる。
   それらの内容を基準版の事実として引用せず、使う場合は「checkout上の内容」と明記する。
2. 必要に応じてコードベースのファイルを読む。起点は `AGENTS.md`、`.agents/README.md`（存在すれば）、各ディレクトリのREADME、
   `knowledge/summary.md`。行数の多いファイル・期間内のcommit・CI失敗・open issue/PRは深掘りの手がかりになる。
3. 提案を作る。対象領域は**絞らない**。少なくとも次の観点はすべて検討し、根拠がある場合だけ提案にする（水増ししない）:
   - コード負債（重複・死蔵・一貫しない設計・肥大ファイルの分割）
   - ドキュメント / AI環境のドリフト（README・AGENTS.md・skill・memoryと実装の食い違い、二重管理）
   - アーキテクチャの抜本的改善（局所修正では解けない構造的な問題）
   - 非効率（開発フロー・学習queue・CI・データ処理の無駄）
   - memoryの陳腐化・削除候補（共有memoryが導入済みなら、反証済み・期限切れ・固定概念化したエントリ）
   - 制御ルールの改訂（AGENTS.md・skill・`.agents/ops/**`・**このプロンプト自体** `.agents/ops/ops-report/prompt.md`）
   - `ai-proposed` ラベルissueの整理（重複統合・close候補・優先度づけ）
   - 掃除候補（掃除モジュールの出力があればそれを解釈。未導入ならその旨を踏まえる）
   - 研究の次の一手（`knowledge/` の到達点・保留事項から）
   - その他の能動的な助言
4. 既存のopen issue（`github.open_issues.items` に全件を `#番号 [ラベル] タイトル（N日更新なし）` 形式で収録）で既に扱われている事項は、新規提案にせず「#番号 を優先すべき」等の形で参照する。
   前回レポートで未選択だった提案を再掲する場合は、新しい根拠があるときだけにし、その旨を書く。
5. `ops_memory.files` は専用memoryの **origin/main版**（`ops_memory.base_sha`）。checkout上の専用memoryよりこちらを優先する。
   要約→索引→関係する観察を読み、前回の未解決点と再検証条件から調査を進める。分担する場合、必要な部分だけを子へ渡す。
   `ops_memory.persistence` に未マージ/失敗/中断があれば、その更新を有効な観察とみなさない。
   人間の採否は `previous_reports.reports[].proposals` と `ops_memory.decisions` を参照する。
   未チェックは見送りの意思表示ではない。明示的な見送り理由は `decisions` 等の根拠を使い、推測で作らない。
   調査の継続に必要な観察を `memory_update` に返す。子は書かず、親スクリプトが公開成功後にだけ保存する。
   issueの進捗・提案本文を複製せず参照にする。共有5層は更新しない（保守は #1192 の責務）。
   再検証・圧縮・削除は memory skill の「運用レポート専用memory」に従う。

## 品質基準

- 各提案は単独で子issueになっても作業を始められる具体性を持つこと（対象ファイル・issue番号・期待効果・完了条件の目安）。
- 根拠（`evidence`）にはファイルパス・行数・issue/PR番号・commit等の検証可能な事実を書く。推測は推測と明記する。
- 選択肢が本質的に複数ある提案は `options`（2〜5個）で示してよい。人間はそのうち1つを選ぶ。
- 提案数の目安は 5〜15件。優先度 `high` は次の予定回までに着手すべきものに限る。
- すべて日本語で書く。

## 放置PR

次のPRは一定期間更新がありません。各PRの現状（差分規模・変更ファイル・mergeable・CI・関連issue・本文）は
収集データの `github.open_prs[].detail` にあります（PRのbranchの内容はcheckoutに無い）。それをもとに、
「継続 / rebaseして仕上げ / close」のどれを推奨するか（`recommendation`）と理由（`reason`、1〜3文）を
`stale_pr_notes` に書いてください。キーはPR番号の文字列で、下の放置PRすべてについて**ちょうど1件ずつ**必要です
（放置PRが無ければ `{}`）。放置PR自体を `proposals` に重複して入れないでください。

{{STALE_PRS}}

## 出力形式（厳守）

最終メッセージは**次のJSONオブジェクト1つだけ**にしてください（前後に文章を付けない。付ける場合も ```json フェンスを1つだけ）。
スクリプトがこのJSONを検証し、IDを付けてissue本文を組み立てます。形式違反は実行失敗になります。

```
{
  "summary": "今回の集計期間の要点を3〜8行のMarkdownで（箇条書き可。チェックボックスとHTMLコメントは使わない）",
  "proposals": [
    {
      "category": "下のカテゴリキーのいずれか",
      "priority": "high | medium | low",
      "title": "1行の提案タイトル",
      "detail": "何を・なぜ・どうするか（1〜6行のMarkdown。チェックボックスとHTMLコメントは使わない）",
      "evidence": "根拠（ファイルパス・issue/PR番号・数値）",
      "options": ["任意。2〜5個の選択肢"]
    }
  ],
  "stale_pr_notes": {"123": {"recommendation": "継続 | rebaseして仕上げ | close", "reason": "理由（1〜3文）"}},
  "memory_update": {
    "summary": "専用memoryの新しい要約（Markdown）。索引へのリンクはスクリプトが付ける",
    "upsert": [{
      "slug": "ci-rerun-observation",
      "title": "CI再実行の待ち時間を再検証する",
      "focus": "CI/依存",
      "observation": "根拠から確認できた観察だけを書く",
      "evidence": "検証方法・ファイルパス・PR番号等",
      "hypothesis": "未解決仮説。なければ空文字",
      "deferred_reason": "人間の見送り理由と根拠。なければ空文字",
      "recheck_when": "次回、CI構成が変わったら待ち時間を比較する",
      "references": ["レポート/提案ID/issue等への参照"],
      "last_verified": "2026-10-18"
    }],
    "delete": [{"slug": "obsolete-observation", "reason": "反証・重複・仕様への昇格等の理由"}]
  }
}
```

`memory_update` は必須。再利用できる観察や変更がなければ `null`。
変更時は `summary`・`upsert`・`delete` の3キーをすべて返す（配列は空でよい）。未指定の観察は維持される。
`slug` は小文字英数字のkebab-case（`index` / `summary` は予約）。新規エントリの作成日は最終確認日を使い、
既存の作成日は維持する。`last_verified` は実際に確かめた日で、未来の日や以前の確認日より前には戻さない。
上のJSONは書式例なので、存在しない削除対象・未確認の観察・例の日付をそのまま返さない。

カテゴリキー:

{{CATEGORIES}}

## 収集データ

```json
{{COLLECTED_JSON}}
```
