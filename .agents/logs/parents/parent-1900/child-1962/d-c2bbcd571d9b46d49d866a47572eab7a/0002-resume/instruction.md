調査報告を受けた親の判断です。この内容で実装し、PRを作成してください（PR本文は日本語、`Closes #1962` を明記）。

## 採用する方針
1. 集計: 全workerターンとイベントの和集合からターンを列挙する純粋処理を切り出す。対応キーは報告どおり (repo.casefold(), parent, issue, delegation_key, turn.sequence)。新形式イベントのない親も登録し、イベントが一切ない入力でもREADMEを生成する。固定main c807cd2a の 51／4／47 と、イベントだけのターン（#1816 ターン2・3）を含む和集合 53 を、分母を分けてREADMEに明記する。「導入前」と断定せず、報告の文言（対応イベントなし。導入時期・未捕捉・捕捉失敗・未公開の区別は未確認）にする。
2. 型付き観測: 公開用の型付き観測（execution・final・publication・termination）を、既存v1イベントを変えずに版付きの拡張として追加する。READMEの生成は公開snapshotだけを読み、private journal を直接読まない。private state を読むのは捕捉段階だけ。workerログ捕捉で公開成功と final の有無を分離する。4状態（最終報告なし・根拠あり／稼働中／終了未確認／最終報告あり・公開保留）を報告の根拠表どおりに区別し、終了記録だけ・awaiting_final だけで欠落と認定しない。重複排除はターンキーで行う。
3. 公開の所在: publication 観測（main / public_pr / pending / unknown、stage・error・PR URL・PR HEAD・照合先main revision）を捕捉・表示する。PR上のファイルをmain入力へ混ぜない。#1862 のターン2・3から PR #1884・#1889、ターン4からmainの3ファイルへ辿れることをテストで示す。PR merge後の再生成は既存の `publish-decision-log --retry` の後着ファイル修復経路を使い、再生成までは「観測時点の公開PR」と表示する。
4. 過去事例: #1816 ターン2・3、#1796、#1864 について、既存の明示コメント（報告で挙げたURL）を出典とする観測を限定的に追記する入口またはデータを用意する。#1814 は途中報告と完了報告の両方を残して「要照合」のまま。任意のsummary文字列を型付き根拠へ昇格させない。
5. 終了（項目4）: termination 観測の schema・表示・テスト（「公開後に終了」の順序、親ディレクトリのない親、引継ぎ、公開中断と再照合）をこのPRで作る。値の取得元は兄弟の子 #1961 が作る次の契約に合わせる:
   - 読取り専用API `remaining_publications(repo, parent, *, state_dir)` が、未完了の公開の一覧（kind: worker_log / decision_log、parent、issue、public_key、sequence、state、stage、error、pr_number、pr_head。decision_log は未公開件数・batch識別）を返す。
   - release / handoff / supervisor終了の時点で、privateな終了記録 {repo, parent, generation, event: ended|handoff_pending|handoff_accepted, observed_at, main_revision, remaining, publication_actor, resume_command} が保存される。
   この契約の名前・位置は #1961 のPRで確定する。今は上の形のfixtureでテストし、捕捉側の読み込みは差し替えやすい1か所に閉じる。parent/cli.py の release・handoff への配線はこのターンでは行わない（#1961 のmerge後に、親が同じworkerへ接続を依頼する）。
6. 仕様README（英日中の3言語）を更新する。

## 検証と終了条件
- 報告のテスト表どおり既存テストを拡張し、固定mainのmanifest方式で51／4／47を固定する回帰テストを入れる（ネットワーク・過去git履歴に依存しない）。既存の公開policy・presentationの負例を維持する。
- pytest は decision_logs 関連のテストと tests/e2e/agents_ops を実行し、pre-commit（ruff・mypy）を通す。
- 作業時間の上限は6時間。1〜6 を実装してPRを作成したら終了する。途中で止めてよいのは、方針と矛盾する事実が見つかった場合と、6時間を超えた場合だけ。止める場合は、済んだ項目・残りの項目・理由を報告する。
- 最終報告には、PR番号、項目ごとのテスト名、5 で仮定した契約と #1961 接続時に必要な変更、未解決の点を書く。
