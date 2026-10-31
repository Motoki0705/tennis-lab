# Herdrの親担当表示

Herdr 0.9.3の表示用metadataへ、parent registryの担当を投影する。
通常の呼出し手順は[herdr-session-label skill](../../skills/herdr-session-label/SKILL.md)。
担当・合意・引継ぎの正本は[parent](../parent/README.md)。

## 表示と同期

上段は`Alice · #1353 · 親`、下段は`セッション表示 · claude`。
名前は[board](../board/README.md#名前とライフサイクル)の有効な割当を読み取り専用で参照する。
名前のない親（`name_status: unassigned`）は従来の`#1353 · 親`を表示し、名前を推測・割当しない。
DBの破損や未知schemaはエラーにし、未割当として隠さない。
`display_agent`は表示名だけを変更し、実際のagent種別と稼働状態には触れない。
source `tennis-parent`で`tl_name` / `tl_parent` / `tl_topic` / `tl_engine` / `tl_owner`を管理する。
`tl_name`は有効な名前だけを持ち、未割当では削除する。
`tl_owner`はセッション識別子のSHA-256で、サイドバーには表示しない。
短い題名は親issueタイトルの識別名。`--topic`で明示変更でき、同じ担当の同期では保持する。
日本語の表示幅を数えて最大20桁に短縮する。

`parent register/reconcile/sync/handoff/release`の担当操作後に、registryのロックを離し、
boardの名前同期→担当表示→担当復帰時の未読表示の順で同期する。
名前同期と担当表示は、一方が失敗しても他方を試みる。
register・handoffの受領時に、担当sessionと確認済みterminalの対応をprivate stateへ保存する。
この対応は表示トークンを消しても終了hook後も保持する。
owner記録がない場合、通常のsyncや起動hookは表示先を確認せず、明示bindを求める。
確認済み記録だけは、[ClaudeのSessionStart](#claudeのsessionstartによる表示先の移動)で条件付きで移せる。
担当操作が成功して表示だけ失敗したときは終了値1、`parent_action_applied: true`と
`display_sync.status: error`を返す。担当操作を取り消さず、表示CLIで修復する。
Herdr環境・bindのないheadless実行は`not_configured`を明示し、担当操作自体は成功を返す。
表示CLIでの`not_configured`は終了値1。設定・API・読み戻しの不一致は終了値2。

`.claude/settings.json`のSessionStart/SessionEnd hookは、Claudeが渡す実session_idを用いる。
hookがそのpaneのforeground agentの子孫であることも確認する。
途中に別の同種agentがある場合は子のhookとして拒否し、親の表示を変更しない。
SessionStartは`CLAUDE_ENV_FILE`へ識別子を追記し、担当済みなら表示を復元する。
ただし`source: clear`（Claudeの`/clear`）では表示を解除し、同じsession IDに担当が残っていても復元しない。
`source: compact`（手動・自動のコンテキスト圧縮）は作業の継続なので、Herdrへの同期を行わず
現在の表示・表示先の対応を保持する。親・関連・未登録のいずれも再命名・解除しない。
新しいSessionStartがHerdr内から届いた場合は、そのpaneに残った前セッションの表示も解除・更新する。
通常のsync/clearは別セッションの`tl_owner`を見つけたら拒否する。
終了・引継ぎ待ち・未担当では自分が付けた表示だけを解除する。hookは担当の生死を推測せず、
registryやGitHubを書き換えない。Codexも同じparent CLIとskillを使う。
解除時は`tl_owner`のない初回設定の「関連」「親未登録」も、管理対象の`tl_*`トークンで識別して消す。
管理対象トークンのない別用途の表示名と、他のトークンは保持する。

未読表示の正本と通常の更新は[boardの未読表示](../board/README.md#herdrの未読表示)を参照する。
通常の担当表示syncは`tl_inbox` / `tl_inbox_label`を保持する。
表示解除と別の会話へのSessionStartでは、同じpaneロック内でsource `tennis-board`の
未読トークンも解除し、前の親の未読数を残さない。DBの未読は変更しない。
同じ担当のresumeでは名前を復元し、未読表示を保持する。compactでは両方とも保持する。

## Codexのclear通知

Codex 0.163.0の標準`SessionStart` hookは、実機では`/clear`直後ではなく次の入力時に実行される。
共通daemonで動く場合はHerdrの端末環境も継承しないため、このhookを表示解除には使わない。
同版へ適用する[TUI側のパッチ](codex-0.163.0-terminal-clear.patch)を同梱する。
新しい会話への切り替えが成功した後、次の入力を処理する前に、端末側から一度だけ通知する。
clearに失敗した場合、`/compact`、自動圧縮、単なる画面消去では通知しない。

起動元で`CODEX_TUI_SESSION_CLEAR_COMMAND`へ、実行可能な`codex_terminal_clear.py`の絶対パスを指定する。
設定値は単一の実行ファイルで、引数・シェル式・repo由来のhook設定として解釈しない。
通常のCodex hookの承認状態は変更しない。

```bash
CODEX_TUI_SESSION_CLEAR_COMMAND="$(realpath .agents/ops/herdr_labels/codex_terminal_clear.py)" \
  /absolute/path/to/patched/codex
```

通知はstdinの`{"version":1,"event":"clear","session_id":"新しいroot threadのUUID"}`。
callbackは継承したHerdr paneとforeground Codexの子孫関係を検証し、管理表示だけを解除する。
旧会話から遅れて届くsyncを拒否し、新しい会話の表示先をprivate stateへ保存する。
その後はHerdr環境のないdaemon上のツールからも、通常のparent CLIで自分の表示を更新できる。
担当registryの割当は変更しない。Herdr外では`not_configured`を返して何もしない。

callbackの非ゼロ終了・起動失敗・10秒のtimeoutはCodex画面にエラーを表示する。
Python側の直近の失敗理由はprivate stateの`herdr-labels/terminal-clear-error.json`に記録する。
失敗を放置せず接続先・プロセス・設定を確認する。フォーカスや題名による代替先の推測は行わない。

この変更が必要なのはTUIクライアントで、共通daemonの再起動は不要。
起動済みTUIにはバイナリの差し替えだけでは反映されない。稼働中の会話を勝手に終了せず、
新規起動から適用するか、会話を保存して再開するタイミングを人間と決める。
Codexの更新時は、その版のclear経路を再確認してパッチ・実機検証を更新する。

### パッチの再現

Codexの別worktreeを`rust-v0.163.0`から作り、パッチを適用する。
既存のCodex checkoutの変更を上書きしない。Rust toolchainとシステム依存は同版の`docs/install.md`に従う。
このリリースtagはCargoのworkspace versionとlock内のローカルpackage versionが異なるため、
パッチにはこの版表記の正規化も含む。外部依存のversion・checksumは変更しない。

```bash
# Codexの専用worktree内
git apply /absolute/path/to/tennis-lab/.agents/ops/herdr_labels/codex-0.163.0-terminal-clear.patch
cd codex-rs
RUST_MIN_STACK=8388608 cargo test -p codex-tui --profile dev-small --locked -- --test-threads=2
cargo build -p codex-cli --profile dev-small --locked
```

TUIテストはclear成功・失敗、通知失敗、new/compactの保持を検証する。
実機でも共通daemonへ接続した使い捨ての会話で、次の入力前の解除を確認する。

## 同期時の照合

registryはatomic renameで保存されるJSONを読む。取得中に担当が変わったら更新を拒否する。
Herdrの対象はcaller contextか明示bindだけで、フォーカスや作業ディレクトリから選ばない。
更新直前にterminal/agentを照合し、更新後に全表示項目を読み戻す。同じ表示への再送は無変更。
同じサーバー・terminalの同期と起動・終了を専用ロックで直列化する。親registryのロックは使わない。
foreground PID・開始時刻・paneの現在sessionを更新前後に確認し、起動・終了後の古い処理を拒否する。
準備中に表示名や管理対象トークンが変わった場合も拒否する。
トークンにはTTLを付けず、終了・再開のイベントで解除する。定期ポーリングは作らない。

## ClaudeのSessionStartによる表示先の移動

Claudeの`startup` / `resume`で、registryのactiveな担当本人が別の端末へ戻った場合に限り、
確認済みowner記録を新しい端末へ移してから表示を同期する。
`HERDR_ENV=1`、実payloadのsession ID、継承したpaneのforeground agentの子孫検査が必要。
新paneにnative参照`agent_session`がある場合は、呼出し元のClaude sessionとの一致も必須。
フォーカス・題名・作業ディレクトリから移動先を選ばない。

旧記録のscopeと`session list`の稼働サーバーを照合し、そのサーバーのagent一覧から旧terminalを探す。
旧terminalがない、別の会話・プロセスへ置換済み、旧サーバー停止の場合だけ、空いたと判断する。
旧terminalに呼出し元のactiveなpanes記録がありforegroundプロセスのstampが一致する場合、
または旧terminalの`tl_owner`が呼出し元のtokenの場合は、生存中として拒否する。
Herdrの取得失敗・不正応答・曖昧な対応も確認不能として拒否し、空きとはみなさない。
ownerのロック内で旧記録を読み直し、確認時と同じ場合だけ置き換える。同時に別の開始が移動した場合は拒否する。

記録のない初回設定、通常のsync / clear / bind、parent CLIの担当操作には自動移動を適用しない。
`clear` / `compact`とSessionEndの扱いも変えない。
SessionStartの拒否・失敗は、`herdr_label.status: error`と理由をstdoutのJSONへ出し、
会話向けの`hookSpecificOutput.additionalContext`（`hookEventName: SessionStart`）と
人間向けの`systemMessage`で修復手順を案内する。
会話にJSONを届けるため終了値は0とするが、通知内では表示同期の失敗と明示する。
SessionEndの失敗は従来どおりstderrと終了値1で返す。

## Herdrの表示設定

人間がユーザーのHerdr設定で、既存項目を残してAgent行を次にする。コードは設定ファイルを書き換えない。
編集前に設定を退避し、別のrows_by_agent設定があれば競合を確認する。

```toml
[ui.sidebar.agents]
rows = [
  ["state_icon", "agent"],
  ["$tl_topic", "$tl_engine"],
  ["$tl_inbox_label"],
]
```

名前は`agent`の表示名に含まれるため、`$tl_name`を重ねて配置する必要はない。
未読は専用行にし、長い題名で隠れにくくする。0件ではラベルがなく、行も消える。
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
通常のsyncはagentが終了・置換されたら拒否し、別paneへ黙って追従しない。
保存先は`${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/herdr-labels/`。
`bindings/`がheadless実行の接続先、`owners/`が担当sessionとterminalの対応、`panes/`が表示同期の現在session。
手動で表示先を変える場合は、旧画面の終了・実担当・新しい会話を確認し、該当するbindとowner記録を
退避してからbindを明示実行する。SessionEndによる終了後はSessionStartか確認済みbindで再開する。

今回だけの全セッション移行は通常のself-service CLIと分け、親が確認済みpaneへ表示用APIを使う。
確認済みの親画面のowner対応も初回移行で保存する。関連画面には親の対応を登録しない。
共有session IDで複数画面が開かれている場合、IDの一致だけで全画面を親扱いにしない。
会話で役割を確認し、関連作業は`関連`、担当未登録は`親未登録`と表示する。
自動同期は永続したowner対応を必須とする。別画面での再開は上記のSessionStartの条件で照合し、
生存中の担当と同じsession IDを持つだけの関連画面を親へ自動昇格させない。

## 検証

```bash
.venv/bin/python -m pytest tests/e2e/agents_ops/test_herdr_labels.py -n 4 -q
```

公式仕様: [metadata API](https://herdr.dev/docs/cli-reference/#panes)、
[サイドバー](https://herdr.dev/docs/configuration/#sidebar-row-layouts)、
[Claude hooks](https://code.claude.com/docs/en/hooks)。
