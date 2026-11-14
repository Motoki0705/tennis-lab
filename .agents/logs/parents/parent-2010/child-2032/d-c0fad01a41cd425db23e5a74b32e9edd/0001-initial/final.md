設計・根拠・テスト案を[#2032の調査報告](https://github.com/Motoki0705/tennis-lab/issues/2032#issuecomment-6880129367)へ記録しました。コード変更・commit・PR作成はありません。

**推奨は、配送以外を先行し、配送には案Aを採用する進め方です。**

| 配送fence案 | 具体的に起きること | 変更・検証 |
|---|---|---|
| **A：ホスト共通fence（推奨）** | Blakeへの配送中は、BlakeのhandoffもCaseyのreleaseも待つ。担当を変えないregistry操作は進む | 全担当変更入口が同じ排他fenceに参加。各入口の待機中にも第三プロセスがregistryを取得できることを検証 |
| **B：親ごとのfence** | Blakeへの配送中もCaseyのreleaseは進む。明示的な再送では送信者も固定する | 複数キーの順序取得と対象集合の再照合が必要。無関係な担当変更の並行性とdeadlock防止も検証 |

両案とも、parentのregister／handoff／release、dispatcherの予約・受理・起動前取消・完了、期限切れ解除とjournal復旧が参加します。特に復旧は`Store.save()`を通らないため、保存ガードだけでは不足します。dispatcher部分は#2031との担当調整が必要です。

配送は「短いregistry/DB照合 → uncertaintyをcommit → registry/DBを離して通知 → 再照合して結果保存」とし、その間はfenceで担当変更を待たせます。受理不明時は再送不可を保持します。配送が先なら旧担当への試行を終えてからhandoffし、handoffが先なら変更後の担当を照合します。受理済みCodex queueがhandoff後に旧threadで始まることまでは防ぎません。

fence判断なしで先行できる範囲は次です。

- **Project通信**：親・Projectのoperation lockを保持し、mutation／REST POSTをregistryの外へ出す。各観測をfreshにし、担当・issue/PR・Projectを前後照合する。
- **Herdr表示**：表示操作を直列化し、固定したterminal・owner・generationで更新・readbackする。失敗は表示結果へ返し、投稿や既読を保持する。
- **Codex探索**：経路無効化・探索・bind保存を分け、走査をロック外で行う。generationと探索epochで古い探索結果の上書きを拒否する。

先行PRでは既存のregistry→routes順を保ちます。routes.lock待ちまで外へ出す変更は、配送fenceのPRでbind／Claude hookとまとめて行います。

PRは **Project I/O／board表示・探索／配送fenceの3本**を推奨します。人間への確認事項は、A/Bの選択と、recover・期限切れ強制解除も配送完了を待つ扱いでよいかです。

検証は既存契約 **37 passed／21.91秒**、追加計測 **4 passed／4.92秒**。計測でもmutation・POST・Herdr・sessions走査のロック保持を確認しました。fence実装、ruff/mypy、新HEADのCIは未実施です。validatorは未指定、試行0／完了0です。

branchは`codex/board-project-io`、HEADは`7a91ddd48070e84f96b80ac6181ba2989607db87`で、worktreeはcleanです。