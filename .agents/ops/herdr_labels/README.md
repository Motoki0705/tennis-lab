# Herdrの親担当表示

Herdr 0.9.3の表示用metadataへ、parent registryの担当を投影する。
通常の呼出し手順は[herdr-session-label skill](../../skills/herdr-session-label/SKILL.md)。
担当・合意・引継ぎの正本は[parent](../parent/README.md)。

## 表示と同期

上段は`#1353 · 親`、下段は`セッション表示 · claude`。
`display_agent`は表示名だけを変更し、実際のagent種別と稼働状態には触れない。
`tl_parent` / `tl_topic` / `tl_engine` / `tl_owner`だけを管理する。
`tl_owner`はセッション識別子のSHA-256で、サイドバーには表示しない。
短い題名は親issueタイトルの識別名。`--topic`で明示変更でき、同じ担当の同期では保持する。
日本語の表示幅を数えて最大20桁に短縮する。

`parent register/reconcile/sync/handoff/release`の担当操作後に、registryのロックを離して同期する。
register・handoffの受領時に、担当sessionと確認済みterminalの対応をprivate stateへ保存する。
この対応は表示トークンを消しても終了hook後も保持する。未確認の表示先をsyncや起動hookだけで推測しない。
担当操作が成功して表示だけ失敗したときは終了値1、`parent_action_applied: true`と
`display_sync.status: error`を返す。担当操作を取り消さず、表示CLIで修復する。
Herdr環境・bindのないheadless実行は`not_configured`を明示し、担当操作自体は成功を返す。
表示CLIでの`not_configured`は終了値1。設定・API・読み戻しの不一致は終了値2。

`.claude/settings.json`のSessionStart/SessionEnd hookは、Claudeが渡す実session_idを用いる。
hookがそのpaneのforeground agentの子孫であることも確認する。
途中に別の同種agentがある場合は子のhookとして拒否し、親の表示を変更しない。
SessionStartは`CLAUDE_ENV_FILE`へ識別子を追記し、担当済みなら表示を復元する。
新しいSessionStartがHerdr内から届いた場合は、そのpaneに残った前セッションの表示も解除・更新する。
通常のsync/clearは別セッションの`tl_owner`を見つけたら拒否する。
終了・引継ぎ待ち・未担当では自分が付けた表示だけを解除する。hookは担当の生死を推測せず、
registryやGitHubを書き換えない。Codexも同じparent CLIとskillを使う。

registryはatomic renameで保存されるJSONを読む。取得中に担当が変わったら更新を拒否する。
Herdrの対象はcaller contextか明示bindだけで、フォーカスや作業ディレクトリから選ばない。
更新直前にterminal/agentを照合し、更新後に全表示項目を読み戻す。同じ表示への再送は無変更。
同じサーバー・terminalの同期と起動・終了を専用ロックで直列化する。親registryのロックは使わない。
foreground PID・開始時刻・paneの現在sessionを更新前後に確認し、起動・終了後の古い処理を拒否する。
準備中に表示名や管理対象トークンが変わった場合も拒否する。
トークンにはTTLを付けず、終了・再開のイベントで解除する。定期ポーリングは作らない。

## Herdrの表示設定

ユーザーのHerdr設定で、既存項目を残してAgent行を次にする。
編集前に設定を退避し、別のrows_by_agent設定があれば競合を確認する。

```toml
[ui.sidebar.agents]
rows = [["state_icon", "agent"], ["$tl_topic", "$tl_engine"]]
```

メタデータのないagentは標準のagent名を表示する。親未登録を親として表示しない。
Space一覧と状態表示は既存設定を使う。
`herdr config check`で検証し、`herdr server reload-config`で再読込する。
実際のサイドバーへの反映を確認する。別クライアントに反映されない場合は、
UIのメニューにある`reload config`でクライアントのローカル設定も再読込する。
設定の有効化のためにサーバーやagentを停止しない。

## Herdr環境変数が届かない実行基盤

人間から表示設定・修復を任された親が、`herdr session list --json`と対象を明示した
`agent list` / `pane read` / `pane get`で**自分の会話**を確認してから実行する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m herdr_labels bind \
  --herdr-session default --pane <確認したpane_id> --terminal <確認したterminal_id>
PYTHONPATH=.agents/ops .venv/bin/python -m herdr_labels sync
```

識別子の入力・上書きは不要で、bind対象は実行中の自分のsessionだけ。
bindは表示先のローカル記録であり、担当を割り当てない。
以後はterminal ID、agent種別、foreground PIDと開始時刻を照合する。
agentが終了・置換されたら拒否し、別paneへ黙って追従しない。
保存先は`${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/herdr-labels/`。
`bindings/`がheadless実行の接続先、`owners/`が担当sessionとterminalの対応、`panes/`が表示同期の現在session。
表示先を変える場合は、旧画面の終了・実担当・新しい会話を確認し、該当するbindとowner記録を
退避してからbindを明示実行する。SessionEndによる終了後はSessionStartか確認済みbindで再開する。

今回だけの全セッション移行は通常のself-service CLIと分け、親が確認済みpaneへ表示用APIを使う。
確認済みの親画面のowner対応も初回移行で保存する。関連画面には親の対応を登録しない。
共有session IDで複数画面が開かれている場合、IDの一致だけで全画面を親扱いにしない。
会話で役割を確認し、関連作業は`関連`、担当未登録は`親未登録`と表示する。
自動同期は永続したowner対応を必須とし、別paneの`tl_owner`も補助的に照合する。
同一Claude sessionを複数画面で再開した場合も、関連作業の画面を親へ自動昇格させない。
表示済みの親画面が終了した後に再開する場合は、残る実担当と表示先を確認して同期する。

## 検証

```bash
.venv/bin/python -m pytest tests/e2e/agents_ops/test_herdr_labels.py -n 4 -q
```

公式仕様: [metadata API](https://herdr.dev/docs/cli-reference/#panes)、
[サイドバー](https://herdr.dev/docs/configuration/#sidebar-row-layouts)、
[Claude hooks](https://code.claude.com/docs/en/hooks)。
