# #2040 の実装（PR作成まで）

担当Issue #2040 の本文が要件と受入条件の正本です。#2029（PR #2038）の transaction用/ロック外 client の分離（`_require_scope`、`io_scope`、`detached()`、cache の transaction epoch、`fresh_reads`）と、#2030（PR #2035）のラベル限定探索が main に入っています。通信方式を差し替えても、これらの契約（ロック解放の範囲、cache の寿命、鮮度確認）を変えないでください。

## 進め方

- 最初に main の HEAD で `PARENT_METRICS_FILE` を使ってベースラインを測る（status / register相当 / note / delegate / review-request など、fake gh で再現できる代表コマンドと、可能なら実GitHubでの読み取り専用コマンド）。実GitHubでの計測は API 上限（5000回/時、並行する親と共有）を消費するので、読み取り専用で最小限にする。
- 通信層の置き換えでは、既存の `GitHubHTTPError` / `GitHubTransportError`、HTTP status の扱い、接続断だけの再試行、ページ重複の取り直し、エラーメッセージ形式、fake gh を使う既存テストの注入境界を保つ。テストの fake を置き換える必要があれば、本番の HTTP 経路を通る fake サーバーなどで同等以上に検証する。
- 認証トークンはログ・例外・metrics・出力へ出さない。そのテストを追加する。
- `--repo` のローカル解決は、remote が一意の GitHub repo に決まるときだけ使い、決まらなければ明示エラーにする（黙って別の repo を選ばない）。
- 並行して #2031（dispatcher・triage）と #2032（board・Project）が進みます。通信層以外の変更は最小限にし、衝突したら後からmergeする側がrebaseします。
- 規模が大きい場合、(a) 通信層の置き換えと `--repo` 解決、(b) 重複呼出の削減、を別PRに分けてよい。その場合は (a) から出す。

## 完了の条件

- 受入条件のテストを含むPRを `Closes #2040` 付きで作成し（分ける場合、最後のPRに Closes、先行PRは References）、関連テストと ruff/mypy（pre-commit）が通り、PRの対象HEADでCI（Python）が成功し、改修前後の計測表がPR本文にあること。
- 時間の上限は8時間。止めてよいのは、(1) 既存契約の変更が必要で親の判断が要る、(2) 8時間を超えた、のどちらかだけ。止める場合は、済んだ範囲・未了の受入条件・次の一手を報告する。
