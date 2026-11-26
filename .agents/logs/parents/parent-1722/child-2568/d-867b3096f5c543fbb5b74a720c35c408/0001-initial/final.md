#2568の規則検出器を実装しました。入口は `detect_scene_events(scene)` と `.venv/bin/python -m src.tennis_scene.ball_events` です。打球・バウンド・ネット接触を元フレーム番号で共通JSONに出力し、短い欠測を挟む候補には `spans_ball_gap` を付けます。ダブルスでは4人の有効な手首・身体位置を使い、打球候補にクリップ内track IDを記録します。 [利用手順](https://github.com/Motoki0705/tennis-lab/blob/aac549c1181e8e34105a00d54fabe8f93f8ed700/src/tennis_scene/ball_events/README.md)

欠測と座標雑音を加えた、調整用とは別の合成テストの結果です。60 fpsで±3フレーム以内を1対1照合しました。

| 種別 | 適合率 | 再現率 | F1 | 対応した候補の時刻MAE |
|---|---:|---:|---:|---:|
| 打球 | 1.000 | 0.717 | 0.835 | 8.16 ms |
| バウンド | 0.998 | 0.876 | 0.933 | 10.26 ms |
| ネット接触 | 1.000 | 0.750 | 0.857 | 6.25 ms |

欠測上限0.34秒、外挿位置の許容差0.55 m、ボールのみの打球の水平速度変化7.5 m/sを含む設定を採用しました。調整セットで13設定を比較し、3クラス平均F1は初期設定の0.623から0.844に上がりました。合成評価はボールのみで行いました。選手近傍を使う打球の精度と実映像の精度は未評価です。 [合成評価記録]([local path])

新規10件を含むテスト41件とpre-commitが通り、ナレッジ再現実行では指標JSONがバイト一致しました。実装branchは `codex/parent-1722-issue-2568-primary`、commitは `aac549c1181e8e34105a00d54fabe8f93f8ed700` です。記録branchは `codex/parent-1722-issue-2568-knowledge`、commitは `643b34abd7a07b016f617aef54cee3909872f127` です。

`clip_012` の先行実行は成功しました。全69クリップの処理と出力照合はローカルジョブで実行中です。完了結果を受けたターンで全件を確認し、実装PRとナレッジ記録PRを作成します。 [全69件の進捗JSON]([local path])

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261125T175340Z_7daad550bc97_meiji-rule-ball-events"]}}