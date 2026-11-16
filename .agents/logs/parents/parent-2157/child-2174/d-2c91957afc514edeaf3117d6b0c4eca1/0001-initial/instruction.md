# #2174 条件付きGETとAPI上限の待機

担当Issue #2174 の本文（目的・変更内容・対象外・Acceptance checklist）を正本として実装してください。親は #2157 です。#2157 の 2026-11-16 のコメントで、CI待機は60秒間隔・上限6時間と決まっています（待機ループ自体は #2176 の担当で、ここでは取得層だけを作る）。

設計調査で確認済みの事実:
- `.agents/ops/issues/transport.py` の `HTTPClient.request`（191行付近）は任意の条件付きヘッダーを受け取れず、258行付近で応答ヘッダーをstatus行だけに縮めている。
- `.agents/ops/issues/cli.py` の `GitHub.get`（320行付近）は200以外をエラーにする。
- 現在のHTTP clientはredirectを拒否するため、jobログ（302で一時URL）を取れない。
- 参考: GitHubの条件付きリクエスト（認証付きの304はprimary rate limitを消費しない）、rate limit（primary: x-ratelimit-remaining/reset、secondary: Retry-After またはreset情報なし）。

要件:
- 新しい取得APIは追加とし、既存の呼出元の挙動は変えない。
- 静かなフォールバックを作らない。保存応答のない304、未知の応答、ログ欠落は理由付きで明示する。
- テストは tests/README.md の方針で設計してから書く。fake HTTPで各応答を再現する。
- 完了したら PR を作成する（本文は日本語、`Closes #2174`）。CIの結果は待たずに報告してよい。
