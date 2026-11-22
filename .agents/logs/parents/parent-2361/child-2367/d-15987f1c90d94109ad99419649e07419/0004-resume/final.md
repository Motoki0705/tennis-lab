#2371 を含む `origin/main` をマージし、`codex/parent-2361-issue-2367-primary` に `2be3b1a49` をpushしました。手動workerのテストは #2371 の版を採用し、暫定fixtureをPR差分から取り下げました。検証結果も追記しました。[PR #2372](https://github.com/Motoki0705/tennis-lab/pull/2372)

- #2371 の既存テストで、固定SHAからの2つのworktreeの起動、個別のE:出力先、重複起動の拒否を確認しました。
- 確認用の子プロセスで実機のD:を隠し、一時repoに複製したopsでも測定の差し替えが効くことを確認しました。20 GiBを1 byte下回ると共通ランナーはCLI起動前に終了75を返し、20 GiBちょうどでは容量判定が許可を返しました。
- 容量保護を含む回帰905件が213.94秒で通過しました。pre-commitとPR差分の `git diff --check` も通過しました。

残りは新しいHEADのCI確認とマージ後の導入です。導入は実行中のCI用キュージョブが完了してから、更新したメインcheckoutで行います。

```bash
sudo systemctl stop tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
spin ops install d-capacity
spin ops on d-capacity
spin ops status d-capacity
cat [local path]
```

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2372,"head":"2be3b1a495d53e1272aae3dfc612c6155efe5ea9","resume_on_success":false}}