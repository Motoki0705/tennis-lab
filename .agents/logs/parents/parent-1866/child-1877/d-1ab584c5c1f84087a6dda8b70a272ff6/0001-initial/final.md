調査を完了し、9項目の詳細を [#1877の設計報告](https://github.com/Motoki0705/tennis-lab/issues/1877#issuecomment-6776601598) に記録しました。コードは変更していません。

1. **候補の正本**  
   元親issueの構造化コメント列を提案します。固定候補ID・版・出典・根拠・切り出し理由・次の作業範囲を保持し、起票・人間判断・実行状態を分離します。既存 `improvement --child` は同じ記録入口へ統合し、親自身も指定可能にします。

2. **CLI入口**  
   `release --ended` と `workflow-report` に共通の終了記録を接続します。候補なしを明示できる形式とし、`improvement-publish`、`improvement-decide`、`improvement-status`、`improvement-reconcile` を追加する案です。supervisorにも完了receiptの照合が必要です。

3. **起票・復旧**  
   issues共通renderで、作成POSTから `issue:parent`[local path]`needs-human` を付けます。専用出典マーカーを追加し、既存のops-report用マーカーは流用しません。triageの送信前記録を共通化し、応答消失後に対象が見つからなければ再POSTせず保留します。既存open・closed・見送り先への集約は、親の明示判断で行います。

4. **承認ゲート**  
   人間発言の引用・出典・対象候補版・対象issue・承認範囲を持つ専用判断イベントを正本にします。register／agree／delegate／各worker入口[local path]、ラベル削除・元親合意・自律根拠では解除できなくします。承認後は既存の担当・runへ接続し、再送や次世代への承認流用を防ぎます。

5. **受入未達の検査**  
   合意した受入項目を固定し、終了時に全項目の充足・根拠・残作業なしを検査します。条件の欠落や候補への付替えは拒否し、既存のmerge・review・worker終了確認も保持します。最終JSONだけのcompletedも拒否します。

6. **判断ログ接続**  
   merge済みの #1865／#1871 のresult・decision・actionとreason参照を再利用します。表示はopenの #1876へ接続します。終了後の別sessionによる記録公開は、現行の担当限定条件との調整が必要です。

7. **検証計画と実測**  
   Acceptance全9項目について、既存nodeの再利用、fake GitHubの故障注入、並行再送、撤回、2世代、表示検証を対応付けました。今回の既存テストは **44件成功、59.02秒**。新機能の受入試験ではありません。

8. **分割案**  
   原則 **1 PR**を推奨します。起票と全入口のゲートを同時に導入する必要があり、工程だけの分割には実益がありません。

9. **親へ返す判断**  
   **#1804との対象解決の分担を確定してください。** 現行mainでは、`issue:parent`・子なしの親は実行対象が空になり、自律完了まで通せません。#1804側の共通resolverを利用する分担を推奨します。人間確認は現行agree同等の信頼境界を想定しています。同じGitHub loginだけでは、発言の捏造まで本人認証できません。

branchは `codex/parent-improvement-loop`、HEADは変更なし、worktreeはcleanです。新commit・PR・pushはありません。