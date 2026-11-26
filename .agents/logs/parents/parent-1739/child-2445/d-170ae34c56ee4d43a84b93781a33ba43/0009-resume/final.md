[PR #2469](https://github.com/Motoki0705/tennis-lab/pull/2469) にダブルス枠の修正と回帰結果をpushしました。40フレームの再現では、連結可能な断片が選手ごとに1枠へまとまり、1選手だけの場合は欠員1になりました。

- Meiji 3cam開発4本の採点値と選別診断は修正前と一致しました。選別後IDF1は84.50%、recallは73.18%です。
- 4cam 69本で各側2 IDを維持し、保存配列1,242組のSHA-256が一致しました。clip_020とclip_058は全フレームで4人を実測しました。
- 関連pytest 41件、最終HEADの追跡unit 12件、pre-commit、knowledge検査が通過しました。回帰結果は既存ノード `000102` に追記しました。

ブランチは `codex/parent-1739-issue-2445-primary`、新HEADは `f8add8f0444ce8581cf3d266f6eee91716b4c5ee` です。CIは結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2469,"head":"f8add8f0444ce8581cf3d266f6eee91716b4c5ee","resume_on_success":false}}