---
title: "公開向け文書は英語版を既定にして日本語版を別ファイルに分け、ハッシュ値は書かない"
applies_to: all
source: "#1318（2026-10-21のユーザー指示と回答）"
created: 2026-10-22
last_verified: 2026-10-22
evidence: "README.md / README.ja.md、docs/installation/README.md / README.ja.md、LICENSES.md / LICENSES.ja.md の冒頭の言語リンク。tests/unit/tennis_scene/test_installation_assets.py"
---

ユーザーは、外部の利用者が読む文書（トップのREADME、導入ガイド、LICENSES、Hugging Faceのcard）で、同じ段落に日本語と英語を並べる書き方を嫌う。英語版を既定のファイル名にし、日本語版は `.ja.md` に分け、冒頭のリンクで切り替える形を選んだ。

また、文書の表にSHA-256などのハッシュ値を載せると冗長になるので書かない。固定値はコード（preflight・取得CLI）だけに置き、文書には「検査する」と書くにとどめる。同じ表を複数の文書に載せるときは、コードから生成してテストで一致を強制する。

**使い方:** 公開向けの文書を新しく書く・直すときは、英語版と日本語版を別ファイルで両方そろえ、言語間で共通の明示アンカーを使う。repo内部の開発者向けREADMEは対象外（日本語のままでよい）。文書にハッシュ値の表を足さない。
