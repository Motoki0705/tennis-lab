前回のCLI登録で設定台帳のハッシュが変わり、旧FP32キャッシュの再現テストに現在の値が混ざっていました。テストを当時のコミットから取得した30ファイルのハッシュに固定し、台帳を編集した場合は編集後の識別子を使うことも検証しました。

CIで失敗した2件を含むキャッシュ関連8件と、球イベント検出11件が通りました。pre-commitも通過しました。`codex/parent-1722-issue-2568-primary` の新HEAD `3d872a3b5d56eb72589a521d70d914b154445b13` をpushし、再検証の証拠をPRに追記しました。 [PR #2569](https://github.com/Motoki0705/tennis-lab/pull/2569)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2569,"head":"3d872a3b5d56eb72589a521d70d914b154445b13","resume_on_success":false}}