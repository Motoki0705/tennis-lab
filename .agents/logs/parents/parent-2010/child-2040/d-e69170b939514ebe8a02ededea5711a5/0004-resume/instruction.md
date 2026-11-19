# PR #2054 の正式レビュー指摘への対応

正式レビュー（gpt-6-astra / xhigh、run parent-2010-b0bea2c4829b427098328f4f0f0f8b68）の判定は request_changes で、major が2件ありました。親は2件目を HEAD f229a1333 で確認しました。`local_repository()` は `git config --get-regexp remote.*.(url|pushurl)` だけを読んでいます。1件目はあなたがコードで確かめてください。

## pr-2054-1（major）: `.agents/ops/parent/github.py:94`

規則: 認証取得のサブプロセスもregistry lock外で実行する

静的読解による指摘です。register・delegate・review CLIなどはStore.locked()内でParentSessionを生成し、ここからHTTPClient.authenticated_login()を呼びます。GH_TOKEN/GITHUB_TOKEN未設定の初回は、get("user")のio_scopeに到達する前に_token()がgh auth tokenを実行するため、最大60秒の待機中も共有registry.lockを保持します。これは#2029のロック分離契約の退行です。token取得自体をロック外へ移し、復帰時の競合検出を維持してください。既存e2eのHTTP bridgeは_tokenを定数に置き換え、認証取得テストもStoreを使わないため、この経路を検出できません。隔離テストは未実行です。

## pr-2054-2（major）: `.agents/ops/issues/cli.py:173`

規則: 実効fetch/push URLからrepoを一意に解決する

静的読解による指摘です。git configでremote.*.url/pushurlだけを読むため、url.*.insteadOfとpushInsteadOfによる書換えを無視します。例えばoriginがhttps://github.com/upstream/project.gitで、pushInsteadOfによりpush先がfork/projectへ変わる設定でも、複数repoとして拒否せずupstream/projectを選び、--repo省略時の書込み先にします。insteadOfで非GitHubへ変わる設定も検出できません。全remoteについてgit remote get-url --allと--push --allによる実効URLを検査するか、未対応の書換え設定を明示的に拒否してください。既存remoteテストにはこの設定軸がなく、対象コードの再現テストは未実行です。

## 依頼

1. 2件とも、コードを読んで成り立つかを確かめる。成り立たない場合は、根拠（該当行と実行経路）を最終報告に書き、変更しない。
2. 成り立つものは PR #2054 の branch（`codex/gh-http-client`）で直す。
   - 1件目: `gh auth token` による token 取得を registry ロックの外で行う。たとえば、Store のロックを取る前（CLI の入口や ParentSession の生成前）に取得する、または io_scope の中で取得する。ロック外で取得した後の競合検出（transaction epoch など）は保つ。token を出力・例外・ログへ漏らさない契約も保つ。
   - 2件目: `--repo` 省略時の解決で、insteadOf・pushInsteadOf を反映した実効の fetch/push URL を全 remote について検査し、すべてが同じ GitHub repo に決まる場合だけ採用する。書換えで別 repo や非 GitHub になる場合は明示エラーにする。実効URLの取得が難しい設定は、黙って無視せず明示的に拒否する。
3. 回帰テストを足す。ロックを持つ状態で GH_TOKEN/GITHUB_TOKEN 未設定のとき、token 取得中に別プロセスが registry ロックを取れることを確かめる。insteadOf・pushInsteadOf で別 repo や非 GitHub に書き換わる remote 設定が明示エラーになることを確かめる。テストは設計してから書く。
4. 関連テストと pre-commit を通し、push して CI の成功を確かめる。PR 本文の検証欄を更新する。

最終報告には、指摘ごとの判断（直した／成り立たない）、変更点、追加したテスト、実行したコマンドと結果、新しい HEAD を書いてください。
