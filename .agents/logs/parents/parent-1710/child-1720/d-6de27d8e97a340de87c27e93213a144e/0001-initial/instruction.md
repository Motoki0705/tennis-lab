## 実行根拠の更新

2026-11-09、人間が親AIに #1710 の実装推進とmergeの判断を委ねた。本文「作業状況・停止事項」の「実装未着手・別途の指示を待つ」はこの指示で解除されている。実装を進めてよい。

## 前提（stacked branch）

- このbranchは #1719 の PR #1793（branch `codex/explain-html-service`、HEAD 8817935）の上に作ってある。#1793 は #1717 の PR #1787（承認済み、mainへの載せ替え中）の上にある。
- #1716（PR #1783）と #1718（PR #1786: 画像・SVG・宣言的グラフ、managed renderer v1/v2、`parsing.StrictHTMLParser`）は main にmerge済み。ただしこのbranchにはまだ #1786 の変更が入っていない。#1786 のコードは `git show origin/main:<path>` で読める。上流PRの載せ替え後に、親がこのbranchの載せ替えを指示する。
- service（#1719）・store・agy・visualsの実装はこのPRで書き換えない。不足は最終報告に書く（最小の局所拡張が避けられない場合だけ、理由をPR本文に明記）。
- PRのbaseは `codex/explain-html-service` にして #1720 だけの差分を表示する。本文に `Closes #1720` と、#1793 に依存する旨を書く。

## #1719 から引き継ぐUIの要点

- 起動URLのfragmentにある `document_id` と `token` を固定UIで取り込み、tokenはメモリだけに保持して `X-Explain-Token` に付ける。生成説明側（iframe）へtokenを渡さない。
- 状態はrequest IDによるポーリングで取る。進行中は旧版を表示し続け、`succeeded` で更新し、`failed` / `cancelled` / `interrupted` では公開エラーを表示する。
- HTTP入口: `D = [local path]` に対して GET `D`・`D/revisions`・`D/revisions/{revision_id}`・`D/requests/{request_id}`、POST `D/requests`（202）・`D/requests/{request_id}[local path]`・`D/undo`・`D/exports`。公開GETの `D/revisions/{revision_id}[local path]`、固定UIの `[local path]`・`[local path]`。
- 実機の編集1件に約271秒かかる。処理中の表示は、長い待ちでも状況が分かり、読み続けられるものにする（経過時間、cancel可能であることの表示など）。
- 閲覧中の新規画像要求は、人間の判断（2026-11-09）により `image_generation_unavailable` の理由付き未完了になる。UIでは、画像生成は初版作成時だけで閲覧中はできないことと、SVG・グラフ・既存画像の差し替えなら依頼できることが分かるように表示する。

## 今回の作業

#1720 を実装する。テストは偽serviceまたは実service＋偽editorで行い、外部APIを呼ばない。ブラウザ検証は既存のPlaywright/Chromiumの入口（#1786 の `tests/e2e/development/test_javascript.py` と同じ方式）を使う。
WindowsホストのEdgeからの表示と操作（指示送信→処理中の継続閲覧→更新→変更箇所へ移動→undo）を、偽editorで1回実演し、スクリーンショットか短い動画をGit対象外の `outputs/explain-html/` に保存して、パスをPR本文に書く。

## 作業単位と時間の目安

実装・テスト・pre-commit・PR作成・CI結果の確認までを1ターンで完了させる。目安は最大5時間。
止めてよいのは: (1) #1719 の公開契約が #1720 の要件を満たせず、変更が下流全体に影響する (2) iframeの隔離方式がserviceのCSP・origin設計の変更を要する (3) CIが変更と無関係な理由で失敗し続ける。止める場合も成果をcommit・pushし、判断材料を最終報告に書く。

## 最終報告

PR番号・HEAD・CI結果、Acceptance項目ごとのテストと実測、iframeの隔離方式（sandbox属性・origin・postMessageの許可内容）、Windows Edgeでの実演の結果と保存先、上流契約への要望、未確認事項。
