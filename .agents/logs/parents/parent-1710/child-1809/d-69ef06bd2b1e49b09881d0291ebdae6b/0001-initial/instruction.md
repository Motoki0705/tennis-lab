子issue #1809 を実装してください。方針と要件の正本は親issue #1710 と #1809 の本文です。

## 背景と証拠
#1721 の実機確認（PR #1807、未merge）で、指定Gemini（gemini-3.8-flash-high / high）の初版と全体再構成が公開に至りませんでした。実出力・ログ・時間の記録はローカルにあります（読み取り専用。編集・移動しない）:
- [local path]
- [local path] と execution-summary.json（段階別の時間・token・実効model）
- [local path]、*.brief.json、private/ と runtime/ 配下（実出力の候補HTMLと検査エラー）
PR #1807 のbranch codex/explain-html-skill は main より古い base（#1720 の UI branch）に載っています。#1809 は main から作った専用branchで行い、#1807 のコードは参照だけにしてください（skill本体やqualification用scriptの取り込みは #1721 の範囲）。

## 確認済みの不具合
- visuals.py の _css で `behavior\s*:` が `scroll-behavior:` に部分一致して拒否する。
- editor.py の GeminiEditor は timeout=600 を writer と verifier で共有している（remaining()）。全体再構成の writer は600秒を超えた。

## 設計で決めて理由を書くこと
- システム所有部品（CSP meta・managed script）を compiler が生成・置換する境界。writer が同等要素を書いた場合の扱い（拒否か除去か）を黙って補正しない形で決める。
- figure metadata と data-figure-id slot の対応漏れを防ぐ方法（出力schema・prompt・検査エラーを渡す修復ターンのいずれか）。修復ターンを採るなら上限回数・時間を固定し、修復後に全検査をやり直す。
- 数式の表現（MathML 推奨。外部CDN・新規依存なし）。
- 段階ごとの時間上限（実測根拠をREADMEへ）。UIの処理中表示（#1720 / PR #1800 で review 中）と矛盾しない値にする。

## 実機確認
受入の2項目目（3題材の初版と三角測量の全体再構成1件）は、実装と偽agyのテストが通った後に、指定Gemini（agy 経由、--model gemini-3.8-flash-high --effort high）で直列に実行してください。呼出しは合計15回以内。実効model/effort・段階別時間・呼出し数を記録し、成果物は worktree の outputs/ 配下（commit しない）に置き、PR本文に要約を書きます。題材の brief は [local path] を再利用してよいです。並行実行はしないでください（並行時に時間切れとホストのメモリ逼迫を観測済み）。
