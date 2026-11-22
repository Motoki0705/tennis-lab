# #2032 の設計報告（今回はコード変更・PR作成をしない）

担当Issue #2032 の本文が要件の正本です。全数調査: https://github.com/Motoki0705/tennis-lab/issues/2013#issuecomment-6875909177 の B表と、project_progress の行を前提にしてください。#2029（PR #2038）の client 分離は main に入っています。

board の配送は、配送中に担当が変わらないことを今は registry ロックで保証しています。これを外す方式は既存契約を弱め得るので、実装前に人間の確認が必要です。今回は次を報告してください。

1. board 配送の担当固定（fence）の設計案を2〜3案。各案について: 担当変更が起き得る全入口（parent register/handoff/release、dispatcher、期限切れ処理など）がどう参加するか、配送中に担当が変わったとき旧担当への送信をどう扱うか、commit-before-notify と受理不明時の非再送をどう保つか、変更範囲、テスト方法。推奨案とその理由。
2. Herdr 表示更新・Codex sessions 走査・Project の mutation/REST POST をロック外へ出す方法（こちらは fence の判断なしで進められるか）。
3. 1PRで出すか、fence 以外を先に出すかの見立て。

人間が判断できるよう、各案の違いは「何が起きうるか」の具体例で説明してください。時間の上限は2時間。
