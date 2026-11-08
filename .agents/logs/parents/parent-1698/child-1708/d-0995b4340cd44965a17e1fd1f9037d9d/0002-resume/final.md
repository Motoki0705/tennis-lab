`pr-1763-1`を修正し、[PR #1763](https://github.com/Motoki0705/tennis-lab/pull/1763)へpushしました。

- branch: `codex/single-camera-gpt-annotation`
- HEAD: `1bcee87efbbe73f6e8697062c12c75ab221697a9`

handoff・rows・imageに`read_id`を導入し、新しい読取と同一要求の再送を区別しました。編集後は最新状態を取得でき、そのrevisionで次の編集も成功します。

修正前の失敗を再現済みです。追加4件・関連28件が全成功し、pre-commitと[最新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42517556760)もPythonを含め全成功しました。PRの検証欄も更新済みです。

追加判断依頼はありません。指摘の最終確認・merge・merge後main確認を親へ引き継ぎます。