[PR #1829](https://github.com/Motoki0705/tennis-lab/pull/1829) を作成しました。writerの変更申告を廃し、compile後の差分からCandidateを構築します。公開契約は維持しています。

- branch: `codex/explain-html-computed-changes`
- commit: `31421cd15`（tree clean）
- 関連114件・pre-commit成功。unit/integration増分は6ケース。
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42725579736)：Python・Lint・CPU全10レーン成功。
- 実機3候補のreplayで、質問・再構成は意味検査へ、画像要求は `image_generation_unavailable` へ進むことを確認。

局所編集の厳密な機械的範囲保証は従来からありません。今回は追加せず、既存の対象照合・意味検査を維持しました。

実機Gemini確認は未実施です。親の統合レビュー・merge・merge後CIと、#1721での実機確認が残ります。