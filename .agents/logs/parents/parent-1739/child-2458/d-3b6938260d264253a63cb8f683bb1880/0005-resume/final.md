正式レビュー round 1 の修正を `codex/parent-1739-issue-2458-primary` にpushし、PR本文を更新しました。新HEADは `b4b859d2c2876c41873d4186be1b5f0e3ae2858b` です。 [PR #2462](https://github.com/Motoki0705/tennis-lab/pull/2462)

公開・復元の共通処理は、最初のrename前に復旧記録を指す保留ファイルを置き、両参照を読み戻してから解除します。片側の更新後に中断させ、SLCS・レビュー・scene indexが復旧手順付きのエラーを返し、`publish` の再実行または `restore` で回復することを確認しました。#2470 からもこの共通処理を呼び出せます。

キャッシュ互換テストは、Git原本と照合した履歴のハッシュで旧版の互換性を検証し、今回の `dataset_publication.py` の変更が別のハッシュとして残ることを確認しました。

CPUテスト111件（関連103件・20.27秒、キャッシュ互換8件・9.34秒）、pre-commit、knowledge検査751ノードが通過しました。検証は合成データで行い、GitHub CIの結果は親CLIに引き渡します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2462,"head":"b4b859d2c2876c41873d4186be1b5f0e3ae2858b","resume_on_success":false}}