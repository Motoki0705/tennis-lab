# 専用OAuth接続

Google Cloudの専用project/clientを使い、共有rclone OAuthアプリのquotaに依存しない。
Google Cloud ConsoleでDrive APIを有効化し、OAuth同意画面を設定して
「デスクトップアプリ」のclient JSONをダウンロードする。
作成・初回同意は利用者のアカウントで行い、秘密情報を会話へ貼らせない。
手順の正本は [rclone公式](https://rclone.org/drive/#making-your-own-client-id) と
[Google OAuth](https://developers.google.com/identity/protocols/oauth2/native-app)。

JSONとrclone設定はrepository外に置く。例えば:

```bash
.venv/bin/python .agents/skills/tennis-drive/scripts/configure_oauth.py \
  --client-json /private/path/desktop-client.json \
  --config-output "$HOME/.config/tennis-lab/rclone-drive.conf"
export RCLONE_CONFIG="$HOME/.config/tennis-lab/rclone-drive.conf"
rclone config reconnect gdrive: --auto-confirm
```

補助scriptは新規のmode 0600設定だけを作り、既存設定を上書きしない。
tokenはまだ無く、`needs_browser_auth`を認証完了と扱わない。
reconnectの出力にはcredentialが含まれ得るため、AIが実行する場合はstdout/stderrを
mode 0600の一時logへ送る。利用者へはlocalhostの認証URLだけを伝え、raw logを表示しない。
接続後は`drive.py quota`と`inspect . --metadata-only`で権限・既存projectのDrive IDを確認する。

Testing状態のExternal OAuthアプリでは、Drive権限を含むrefresh tokenが7日で失効する。
継続運用ではGoogleの案内に従いPublishing statusも設定し、失効を無限retryで隠さない。
[Googleのtoken失効条件](https://developers.google.com/identity/protocols/oauth2#expiration)

`RCLONE_CONFIG`を設定したprocessからDrive skillとColab `start`を呼ぶ。
既存Colab sessionは開始時に設定を複製済みなので、ローカルの設定変更だけでは更新されない。
実行中jobを終えて保存を確認し、設定を更新したsessionまたは新sessionで接続を検証する。
認証変更のためにdatasetや成果物のDrive path/IDを変えない。
