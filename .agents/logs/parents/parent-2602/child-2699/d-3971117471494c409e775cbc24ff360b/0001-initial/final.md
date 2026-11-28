#2699 の実装を完了し、[PR #2703](https://github.com/Motoki0705/tennis-lab/pull/2703) を ready で作成しました。

- ブランチ `codex/parent-2602-issue-2699-primary`
- HEAD `2ddc1a9bab6986f699ce98256c83b1c859ca5fd2`

固定版の正本から人物モデル比較5ページ分の表25件・グラフ9件を生成し、34組すべてで SVG／PNG、折りたたみテキスト表、正本との照合、再生成時のバイト一致を確認しました。代表画像の目視と全34組の画素照合の結果は PR に記録しています。Wiki 執筆ガイドと全38ページの棚卸しも収録しました。

対象 pytest は45件成功し、ruff・mypy・pre-commit も通過しました。PR CI は起動済みで、完了確認は親のレビュー進行へ引き渡します。公開 Wiki での表示確認は、後続のページ改訂時に行います。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}