親判断が必要です。新方針1を厳密に適用すると、6件とも確認不能で止まります。

同UIDのsystemd・sd-pam・sshd×2・gpg-agent計5件で、cwd・root・exe・fd・maps・map_filesがすべてEACCESでした。読めた範囲では6件への参照は0ですが、この5件の参照可能性を排除できません。

次のどちらにするか確定が必要です。

- 厳密なゲートを維持し、本番に十分な読み取り権限を用意する。
- 関連未観測の確認不能プロセスを診断に留めるなど、除外方針を認める。

[PRへ証拠と判断事項を記録しました](https://github.com/Motoki0705/tennis-lab/pull/2052#issuecomment-6885881242)。本番操作・保護対象68ファイルの変更はありません。

branchは `codex/worktree-cleanup-uncertain-scope`、HEADは旧方式の `44865cd80` のままです。新実装・commit・push・CIは未実施。validatorは指定なし、試行0／完了0回です。