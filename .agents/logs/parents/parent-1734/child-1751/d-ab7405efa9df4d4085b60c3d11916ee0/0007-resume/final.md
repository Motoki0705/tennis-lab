再レビューの1・2・3・5・6を修正し、[PR #2182](https://github.com/Motoki0705/tennis-lab/pull/2182)へ反映しました。

- branch: `codex/scene-inference-runtime`[local path]`9db3a93ad`
- CPU365件、追加13ケース＋既存2ケース拡張、pre-commit成功。
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44287434372)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44287434476)とも初回で全成功。
- 69clip×4cameraの全cache hitをCPUで検証し、モデル生成0・Court metadata読込1回・全量hash1回を確認しました。

4は親判断どおり、runtime宣言後のPython source追加・書換を既知の制限として明記しました。identity更新と再生成の必要範囲もPRに記録済みです。追加GPU実行は0回です。

PRはReadyを維持しています。統合レビュー・merge・closeを親へ返します。