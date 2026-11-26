注釈CLIを設定監査の台帳に登録し、各サブコマンドに共通のパス検証を接続しました。既定設定はリポジトリの共通ルートから解決します。同じブランチに修正をpushし、検証結果を追記しました。 [PR #2448](https://github.com/Motoki0705/tennis-lab/pull/2448)

CIで失敗した2件を含む関連テスト **451件が204.14秒で成功**し、pre-commitも通過しました。固定入力69 clipから再抽出した1,205 frameの選択結果は前回とバイト一致し、3cam・4camの画像パケット生成も確認しました。

ブランチは `codex/parent-1739-issue-2442-primary`、新HEADは `f23c0e0ecd4d208ae0730032548aa46ca10d6f83` です。新HEADのGitHub CI結果を親CLIへ引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2448,"head":"f23c0e0ecd4d208ae0730032548aa46ca10d6f83","resume_on_success":false}}