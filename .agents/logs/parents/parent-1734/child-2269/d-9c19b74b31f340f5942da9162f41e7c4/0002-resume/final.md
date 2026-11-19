PR #2270のレビュー対応を `codex/court-side-player` にpushしました。HEADは `bb3d318e5fc38b007f13620ee91d7adfc9946a5c` です。[PR #2270](https://github.com/Motoki0705/tennis-lab/pull/2270)

- 動画共有では、判定に使った証拠の配列を公開・再利用時に検証します。ボール成果物がないplayer構成で両方が成功し、証拠種別の不一致と選手観測の破損は拒否されました。
- #1757は `court_side.evidence=player` と両ボール機能の `enabled=true` を指定します。本番の選手のみの構成は両方を `false` にするとボール3段が外れ、外した段への個別の `load` は設定時に拒否されます。
- READMEとPR本文に、Hungarian法は同一フレーム内の足元距離だけで対応を求め、`player_association` は判定を受け取る後段であることを明記しました。

保存済みmeiji_4camのCPU再生では、3clipの単独・共有とも `[F,F,T,T]`、停止0件でした。全8仮説のcost・supportが前回と一致し、marginは `0.1734 / 0.2242 / 0.3151`、共有は `0.2372` です。既定のball経路でも基準版の38段を同じ参照で再ロードできました。

ローカル1,502件、#1757との試験マージ上の24件、pre-commitが通過しました。**最終HEADの `Python` CIとKnowledge libraryは成功**しました。レビュー修正の最初のCIは、競合解消のpushでキャンセルされました。[最終CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44464967871)・[Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44464967915)