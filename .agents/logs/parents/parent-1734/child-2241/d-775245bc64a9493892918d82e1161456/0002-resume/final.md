[PR #2244](https://github.com/Motoki0705/tennis-lab/pull/2244) の正式レビュー指摘5件に対応しました。ブランチは `codex/court-side-video-shared`、最終HEADは `65d6c9009` です。親による再レビューに進めます。

- 既定clipでも旧cacheと完成markerが失効する挙動は、親の判断どおり維持しました。baseで保存したstoreとmarkerを使い、再計算・明示overwrite後に同じcourt_side出力を得られることを確認し、PR本文に移行手順を記載しました。
- 旧形式の6閾値だけを持つ設定は、保存値を保ってclipモードで読み込めます。部分的な新形式と、取得方法が未指定の共有設定は拒否します。
- 同じorchestratorでbuild結果を再利用する際も、requestのSHAと全clipの現行入力版を照合します。一括向けAPIでも、証拠clip集合や入力版が異なる準備済み判定を拒否します。
- 共有モードでの `court_side` 実装差し替えは、計画作成時に拒否します。

**関連CPUテスト748件とpre-commitが成功しました。** 保存済み全69 clipのCPU再検証でも `[F,F,T,T]`、margin `0.267333`、一致51・警告18・矛盾0を再現し、全件表と単独診断は初版とbyte一致しました。最終HEADの[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44287378749/job/134755780689)と[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44287380743)も成功しています。

PRはReady for reviewです。validatorは指定なしで、試行・完了ともに0回です。