予測不要の白紙注釈登録API `configure_local_videos()` を実装しました。元動画・frame/PTS・camera/group・公開選択を固定し、冪等登録と入力差替え拒否に対応しています。

- PR: [#1745](https://github.com/Motoki0705/tennis-lab/pull/1745)（レビュー可能・未merge）
- branch: `codex/direct-local-annotation-input`
- HEAD: `df1c7d8a213eb24373e3faa83f13fcc62169e7bd`
- 検証: 重点68件、pre-commit、CI `Python`・Lint・CPU全10 shard成功
- ローカルPSS計測3件の失敗は変更前mainでも再現済み。詳細はPRに記録しました。

実GPT・自動継続・scene CLI・公開writerは後続範囲です。統合レビュー・merge・merge後main CI確認を親へ引き継ぎます。