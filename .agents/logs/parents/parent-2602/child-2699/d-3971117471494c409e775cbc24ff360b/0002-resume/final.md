正式レビューの major 3件を修正し、同じ ready PR に push しました。[PR #2703](https://github.com/Motoki0705/tennis-lab/pull/2703)

- ブランチ `codex/parent-2602-issue-2699-primary`
- HEAD `c615ed793dd8490d0293ad825da8f27f2750d055`

1. Wiki に転記した図表ブロックの全文と SVG／PNG の SHA-256 を照合するようにしました。実正本から再生成した5ページすべてで正常転記が通り、`6,844` → `6,845` の変更と SVG の差し替えは終了コード1で拒否しました。旧スニペットはマーカー付きで再生成してください。
2. 直接参照する PNG にも形式検査を適用し、破損、非RGB、サイズ超過、不要チャンクを拒否することを確認しました。
3. POSIX 絶対パス、Windows ドライブパス、UNC を拒否し、公開 URL・相対パス・単位表記と、実正本5ページのスニペット・Wiki ガイド文書が通ることを確認しました。

対象 pytest は84件成功し、ruff・mypy・pre-commit も通過しました。修正後 HEAD の CI は起動済みで、完了確認は親のレビュー進行へ引き渡します。[CI 実行](https://github.com/Motoki0705/tennis-lab/actions/runs/46046417670)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}