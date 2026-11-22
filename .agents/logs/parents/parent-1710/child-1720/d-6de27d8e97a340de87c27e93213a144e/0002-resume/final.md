[PR #1800](https://github.com/Motoki0705/tennis-lab/pull/1800) をmainへ載せ替え、force-pushしました。**CIは未達です。**

branch: `codex/explain-html-ui`  
HEAD: `a0f2427dbfeb425bb81585aac3a9a5e3ea02a093`

- 旧境界`8817935`からUI自身の3コミットだけをmain `3f11ab019`へrebaseし、検証1コミットを追加。README 3言語の競合はmainのvisual・undo/export説明を保持して解消しました。計4コミット・16ファイル、worktreeはcleanです。
- 実service＋偽editorのブラウザ検証を1シナリオ追加。実renderer v1/v2のslider操作と更新後の入力・描画保持、`sandbox="allow-scripts"`・opaque originでのCSP、`image_generation_unavailable`の理由表示を確認しました。
- ブラウザ5シナリオ成功：33.10秒。関連回帰231件成功：33.72秒。文書検査143件、pre-commitも成功しました。
- service差分は**追加10行**。公開済みrevisionへ選択通知bridgeを付ける閲覧view分岐です。schemaにも`reader.html`の列挙を追加しました。

[CI attempt 3](https://github.com/Motoki0705/tennis-lab/actions/runs/42602776116/attempts/3) はLint・CPU 9本成功、CPU 7・Python集約失敗です。3回とも既存annotationテストのPSS上限超過（1536 MiB）でした。同じテストはローカル単独では成功しています。指定の停止条件に従い再試行を止めました。

証拠は`outputs/explain-html/rebased/`と`rebased-final-ci.json`等に保存し、PR・#1720へ記録済みです。追加確認はChromiumで、Windows Edgeの再実演は未実施です。CI失敗への対応とmerge判断を親へ返します。