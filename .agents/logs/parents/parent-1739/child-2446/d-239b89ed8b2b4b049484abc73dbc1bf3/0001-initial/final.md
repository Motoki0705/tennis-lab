段階2の比較ページを作成しました。初回停止14本を含む16クリップで、旧経路と新経路を同期した4視点とコート上面図で比較できます。両側に同じボール3Dを重ねています。段階2を初期表示にし、クリップと再生位置を保ったまま段階1へ切り替えられます。

公開用の [出力ディレクトリ](https://github.com/Motoki0705/tennis-lab/blob/e118a60835634fdcf445e01d435516a1029e872d/outputs/pose-comparison) に `index.html`、動画32本、ポスター、manifest を保存しました。全動画はH.264で、全体61.68 MB、最大4.19 MBです。

段階2では固定色の時間IDで選手と選外の人物を見分けられます。実測と補間、IDの生成・消失・再出現、割当の接戦も表示します。旧経路に人物3Dがない `clip_020` と `clip_058` では、新経路で全651・765フレームそれぞれに4人分の実測骨格がありました。

pytest 337件、pre-commit、Chromiumでのオフライン再生を確認しました。段階1のMP4は再エンコードで3本のハッシュが変わり、その3本のエンコード前の全描画フレームは元実装と一致しました。

[ドラフトPR #2470](https://github.com/Motoki0705/tennis-lab/pull/2470) を作成しました。ブランチは `codex/parent-1739-issue-2446-primary` です。新recipeの下流接続、既定recipeの回帰確認、3camと全69クリップの定量比較は次の委任に残っています。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2470,"head":"e118a60835634fdcf445e01d435516a1029e872d","resume_on_success":false}}