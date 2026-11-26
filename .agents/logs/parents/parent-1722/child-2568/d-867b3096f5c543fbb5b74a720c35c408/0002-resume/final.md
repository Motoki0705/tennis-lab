multiview-v2の **69件すべてを処理・照合**しました。失敗は0件、処理時間は251.33秒でした。打球425件、バウンド363件、ネット接触35件の候補を保存しました。 [クリップ別の予測JSON]([local path]) · [処理状況]([local path])

検出器の入口は `detect_scene_events(scene)` と `.venv/bin/python -m src.tennis_scene.ball_events` です。欠測付きの調整セットで13設定を比較し、欠測上限0.34秒、外挿の位置差0.55 m、ボールのみの打球で水平速度変化7.5 m/sを含む設定を採用しました。調整セットの3クラス平均F1は0.623から0.844に上がりました。 [閾値一覧](https://github.com/Motoki0705/tennis-lab/blob/aac549c1181e8e34105a00d54fabe8f93f8ed700/src/tennis_scene/ball_events/rules.py)

欠測・雑音付きの独立合成テストを、60 fpsで±3フレーム以内の1対1照合で採点しました。

| 種別 | 適合率 | 再現率 | F1 | 時刻MAE |
|---|---:|---:|---:|---:|
| 打球 | 1.000 | 0.717 | 0.835 | 8.16 ms |
| バウンド | 0.998 | 0.876 | 0.933 | 10.26 ms |
| ネット接触 | 1.000 | 0.750 | 0.857 | 6.25 ms |

新規10件を含むテスト41件とpre-commitが通りました。合成評価は再現実行で指標JSONがバイト一致しました。選手近傍を使う打球の精度と実映像での精度は未評価です。

- 実装は `codex/parent-1722-issue-2568-primary` の `aac549c1181e8e34105a00d54fabe8f93f8ed700` に保存しました。CI実行中です。 [PR #2569](https://github.com/Motoki0705/tennis-lab/pull/2569)
- 合成評価は `codex/parent-1722-issue-2568-knowledge` の `643b34abd7a07b016f617aef54cee3909872f127` に記録しました。Knowledge CI実行中です。 [PR #2570](https://github.com/Motoki0705/tennis-lab/pull/2570)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2569,"head":"aac549c1181e8e34105a00d54fabe8f93f8ed700","resume_on_success":false}}