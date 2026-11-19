# #1990 第1段の追加修正（PR #2023）: 同UIDの検査不能プロセス

報告にあった「同UIDの23件が検査不能」を親が実機で確認した。いずれも `[local path]` の所有者は自分だが、`PR_SET_DUMPABLE=0` によりfd・mapsを読めないプロセスだった。内訳は、Codex CLI本体（agent_runtime.py の子。process hardeningで非dumpable、cmdlineも空）、`sshd: kamimura@...`、`gpg-agent`、`(sd-pam)`。Codexが実行するshellコマンド・pytest・学習は別プロセスで、execによりdumpableに戻るため検査できる。

人間は特権helperではなく「自分のプロセスだけ」を選んだ。親の判断として、次のように実装する。

- 同UIDでも**非dumpable**（ptraceのread権限がなく fd/maps/cwd を読めない）プロセスは、他UIDと同様に検査対象外とする。判定は「EACCES/EPERMで読めない」ことだけに頼らず、非dumpableであることを確かめる根拠（例: `[local path]` 等から得られる情報や、読めなかったファイルとエラー種別の組合せ）を設計して使い、根拠が立たない読取失敗（ENOENT以外の想定外のエラー、dumpableなのに読めない等）は従来どおり失敗にする。プロセスの終了による ENOENT/ESRCH は従来の扱いでよい。
- 除外した同UIDプロセスは、件数と `comm`（読める範囲）を監査記録・ログに残す。READMEの前提（3言語）に「非dumpableな同UIDプロセス（Codex CLI本体、sshd、gpg-agent等）は検査しない。データを読む学習・スクリプトはdumpableなプロセスで動く」ことを書く。
- テスト: 擬似procで、非dumpableの同UIDプロセスが除外されて記録されること、dumpableなのに読めない同UIDプロセスでは失敗すること。
- 実機の読取専用dry-runで、同UIDの検査不能による失敗が消えたことを確認する（配置切替前のため、配置検査による失敗が残るのは構わない。その場合は失敗理由が配置だけであることを示す）。
- PR #2023 を更新し、対象HEADのCIで `Python` 成功まで確認する。

作業時間の上限は2時間。合意の範囲外になる分岐が出たときだけ、選択肢と推奨を書いて止まってよい。
