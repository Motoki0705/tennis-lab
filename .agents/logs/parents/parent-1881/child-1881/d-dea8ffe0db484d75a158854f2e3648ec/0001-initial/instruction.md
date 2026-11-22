# #1881 実装 PR 1: 証拠用Issueへの保存基盤・共通reader・旧コメント読取互換

担当Issue: https://github.com/Motoki0705/tennis-lab/issues/1881 （正式な子なし。本文の方針・完了条件で人間と合意済み）

## 前提（必ず読む）

同じworktreeの `tmp/1881-plan/`（gitignore）に、前のplanning workerの調査資料がある。まず `plan.md`・`inventory.md`・`test-plan.md` を読み、イベント対応表・読取元の file:line・既存テストの棚卸しを再利用する。再調査で重複しない。行番号は HEAD 2ac198a07 時点のもの。

**親（人間の判断を含む）の決定:** planの推奨だった「専用Git ref＋証拠PRのreceipt」は採用しない。代わりに次の方式とする。

- **親ごとに証拠専用のIssueを1つ持ち、機械処理用のマーカー付き記録（`parent-session:v1` / `parent-review:v1` の親側記録 / `issue-start-judgement:v1` など、現在親Issueへ投稿しているもの）はそこへ投稿する。** 既存の「認証publisher（`gh api user` と同じlogin）が書いた先頭マーカーだけを信用する」検証をそのまま使う。
- 親Issueには、人間への質問・合意変更・全体影響・全体の成果だけを残す（これは PR 2 で切り替える）。
- 実装は2PRに分ける。**このターンは PR 1 だけ。** PR 2（全writerの切替、通知判断、親本文の進捗概要、skill/worker指示、6件の再分類、通常入口のe2e）はこのターンでは行わない。

## PR 1 の成果と受入条件

1. **証拠用Issueの作成・特定・検証**
   - 親Pの証拠用Issueを作成/取得する共通入口を作る。作成は冪等にし、再試行や応答喪失で重複作成しない。重複を検出したら選ばずに明示エラーにする。
   - 親との結び付きはGitHub上だけで検証できること: 認証publisherが書いた本文先頭マーカー（例 `<!-- parent-evidence:v1 {"parent":P,...} -->`）、専用ラベル。ローカルregistryに番号をキャッシュしてよいが、読むたびにGitHub上の結び付きと照合する。
   - 開閉状態とラベル名は設計して理由を書く。どの状態でも、作業Issueとして扱われてはいけない。
2. **作業Issueとして扱われないこと**
   - Issueを列挙・処理するすべての経路で、証拠用Issueを明示的に除外する。少なくとも dispatcher（候補選択・判定）、`issue-supplement.yml` を含むGitHub workflow、ops-report、weekly-report、project_progress、worker_status、herdr_labels、issues API、board を確認する。漏れがないことを検索で確かめ、除外した経路の一覧をPRに書く。
   - 証拠用Issueの作成で、補足追加workflowやdispatcherが起動しないことをテストまたは実際の設定で示す。
3. **共通reader**
   - 親Pの機械記録を、**旧来の親Issueコメント**と**証拠用Issueのコメント**の両方から読み、型付きの履歴として返す共通readerを作る。各記録に出典（`legacy_parent_comment` / `evidence_issue`）、comment ID/URL、publisherの検証結果を持たせる。
   - `inventory.md` の読取元（`policy.events`、`parent_review_policy.records`、`judgement_comments`、`agreement_comments`、`execution.instruction_comments`、`review_evidence.followups`、merge gate、review inspection/recovery、proposalsの他親検索、dispatcher collect/guard、supplement、ops-report organization、decision_logs の reason_ref など）を、このreader経由へ切り替える。媒体の選択を各所に散らさない。
   - 同じ論理記録が両方の媒体にある、同じキーで内容が違う、証拠用Issueの結び付きが検証できない、などは明示エラーにする。**静かなフォールバック（片方が無ければもう片方の成功記録を使う等）をしない。** 順序の決め方を明確にし、テストで固定する。
   - 証拠用Issueに付いたマーカーのない通常コメントや、別loginのコメントの扱い（無視せず診断に出すなど）を決めて実装する。親Issue側の通常コメントは、これまでどおり人間の入力として読む（`instruction_comments` の挙動を壊さない）。
4. **既存挙動の維持**
   - PR 1 では writer を切り替えない（投稿先は従来どおり親Issue）。PR 1 単独でmergeしても、既存の親（進行中を含む）の合意・レビュー・merge gate・復旧の判定結果が変わらないこと。
   - 旧形式の記録は期限なしで読めること。欠けたフィールドを現在値で補わない。

## テスト

- [tests README](tests/README.md#テスト作成方針) に従い、`tmp/1881-plan/test-plan.md` の棚卸しを使う。`tests/e2e/agents_ops` の既存テストを再利用し、fake GitHub と使い捨てgitで次を検証する: 両媒体からの読取と順序、旧コメントのみの親で判定が不変、重複・不一致・結び付き不正の拒否、証拠用Issueの冪等作成と重複検出、列挙経路からの除外。
- 変更に関係する既存のagents_opsテストを実行し、件数と所要時間を報告する。ローカルの `tests/e2e/agents_ops` には無関係な既知失敗があり得るので、失敗したら origin/main で同じテストを比べてから退行と判断する。

## 制約

- 実際のGitHubへの書き込み（Issue作成・コメント・ラベル作成・本文編集）は、PRの作成・pushを除いてこのターンでは行わない。証拠用Issueの実作成は PR 2 以降。
- READMEは該当する `.agents/ops/*[local path]` を更新し、同じ事柄を2か所に書かない。
- PR本文は日本語。pushごとにCIが走るので、対象HEADの `Python` の結果を報告する。`[skip ci]` を付けない。
- 作業時間の目安は最大6時間。止めてよいのは次の場合だけ: PR 1 の受入条件をすべて満たしてPRを作りCI結果が出た／親の判断がないと進めない設計上の矛盾が見つかった（その場合も独立部分は進め、コミット済みの状態で報告）[local path]。

## 報告

最終メッセージに、PR番号・HEAD、受入条件ごとの証拠、除外した列挙経路の一覧、テストの増分と実測、CI結果、未確認事項、PR 2 へ持ち越す点、親の判断が必要な点をまとめる。
