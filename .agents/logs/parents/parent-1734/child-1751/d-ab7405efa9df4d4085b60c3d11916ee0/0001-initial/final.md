CPU実装を[draft PR #2182](https://github.com/Motoki0705/tennis-lab/pull/2182)へ提出しました。GPU受入は未完了です。

- branch: `codex/scene-inference-runtime`
- HEAD: `c5fd90a84`、push済み・worktree clean。
- API: `model_specs()`で宣言し、`process_with_models()`内の`borrow()`で借用。runtimeが最終需要node後に解放します。
- 共有キー: 種別・重みSHA-256・architecture/encoder・device・dtype・有効設定。clip/camera IDは除外。
- 検証: 関連367件＋architecture18件合格。全cache/loadで読込0、fake camera数1→5でも各モデル読込1回、例外・中断時の解放を確認。
- 変更範囲のpre-commitと、最新HEADの[CI（必須Python含む）](https://github.com/Motoki0705/tennis-lab/actions/runs/43793067636)は全成功。全repo mypyの未通過はPRへ記録済みです。

残りは配置切替後の実GPU比較と、`model_event=` hookへのC1接続です。C1固定入力のCourt停止・基準値未達も解消が必要です。issueは閉じていません。validatorは指定／試行／完了＝0／0／0です。