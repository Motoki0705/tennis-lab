子issue #1828 を実装してください。方針と要件の正本は親issue #1710 と #1828 の本文です。

## 背景と証拠（読み取り専用。編集・移動しない）
#1721 の実機確認（PR #1807）で、閲覧中の質問（初回）・全体再構成・新規画像要求が、writerの変更申告（Candidate.changes）と実差分の不一致で invalid_model_output になりました。
- [local path]
- [local path]
- [local path]、[local path]、各attemptの private/runtime 配下の候補
原因は、writerがraw HTMLの差分を自己申告し、compilerのslot化（展開済みFigureを空slotへ戻すなど）も含めて完全一致を要求されることです。ContentChange は content_id と kind だけで、editor.py は compile後に content_changes() で再計算して置き換えています。document.validate_candidate が申告と計算の一致を要求しています。

## 設計で決めて理由を書くこと
- writerの出力schema（Candidate とは別の型にするか等）と、runtimeが Candidate を組み立てる位置。store.commit・HTTP・UIが読む Candidate/Revision の changes の意味は維持する。
- evidence_changes の扱い（信頼側contextから決まる部分を申告させない。人間向けの理由文言が必要なら残す）。
- 局所編集の範囲制限が現在どこで保証されているか。計算差分に対して検査されることを確認し、保証が無ければ黙って追加せず報告する。

## 確認
受入の各項目を偽agyのテストで確認します。実機Geminiの呼出しは不要です（#1721 で行います）。可能なら、上記の実機失敗候補を固定入力として使い、修正前に invalid_model_output、修正後に検査へ進むことを示してください（候補をfixtureとしてcommitする場合はprivate情報を含めず、必要最小限にする）。
