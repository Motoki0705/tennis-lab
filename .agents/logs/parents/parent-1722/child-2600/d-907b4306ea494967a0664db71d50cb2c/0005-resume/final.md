正式レビューのmajor 2件を修正し、[PR #2613](https://github.com/Motoki0705/tennis-lab/pull/2613) に push しました。HEAD は `3d8cec367e8a9fb2cb44788bdae3eb494e0aeb3f` です。

- 判定不能区間にある既知イベント15件を教師生成と採点に残しました。`clip_049` の打球は、中心をBCEから除外し、判定可能な f702 の教師値が0.60653になることを確認しました。
- Set Attentionでは、欠員を物理的に除いた集合との出力一致を検証しました。SABとpoolingのmaskを片方ずつ外すと、どちらでも同じテストが失敗しました。修正後は関連36件とpre-commitが通過しました。

最終配布版でCPU前処理を起動しました。初回の予測と重みは元の出力先に保管しています。前処理完了後に、修正HEADのGPUジョブ10件を登録します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T115528Z_4b14423a0aa5_meiji-v3-motion-final-coverage"]}}