[PR #1807](https://github.com/Motoki0705/tennis-lab/pull/1807) をdraftで作成しました。**実装・push・CI確認は完了、実機受入は未達です。**

- branch: `codex/explain-html-skill`
- HEAD: `87d4fff260f2cf5e557418b98c4a77e0bf5dbcd9`
- [最終CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42617987477): Python・Lint・CPU全10本成功
- 最終直列検証: **113 passed / 11.74秒**。pre-commit、ruff/mypy、quick_validateも成功

| 受入項目 | 結果 |
|---|---|
| skill→初版→更新→export | 公開CLI e2e成功。実機でも三角測量の初版・質問・undo・停止後exportが成立 |
| 指定Gemini・全体再構成・画像経路 | 実効設定を確認。再構成と閲覧中画像要求はtimeout等で未成立 |
| 3題材の説明品質 | **未達**。比較・観測誤差は未公開。三角測量にも本文品質の問題が残る |

Geminiは実生成17回＋生成前照合失敗1回。完了14回すべて **Gemini 3.8 Flash (High) / high、agy 1.4.11**でした。完了時間合計3009.37秒、取得usageは1,037,490 tokens。直近3生成のtimeoutを受け、停止条件(1)として追加呼出しを終了しました。

保存した証拠です。

- [評価表・未解決事項](https://github.com/Motoki0705/tennis-lab/blob/87d4fff260f2cf5e557418b98c4a77e0bf5dbcd9/outputs/explain-html/qualification/qualification-status.md)
- [実機呼出集計](https://github.com/Motoki0705/tennis-lab/blob/87d4fff260f2cf5e557418b98c4a77e0bf5dbcd9/outputs/explain-html/qualification/execution-summary.md)
- [質問成功・再構成失敗の44秒動画](https://github.com/Motoki0705/tennis-lab/blob/87d4fff260f2cf5e557418b98c4a77e0bf5dbcd9/outputs/explain-html/qualification/workflow-question-and-failed-restructure.mp4)

初版画像は組込みimagegenで生成・import済みです。[画像](https://github.com/Motoki0705/tennis-lab/blob/87d4fff260f2cf5e557418b98c4a77e0bf5dbcd9/outputs/explain-html/qualification/assets/court-concept.png)と[prompt](https://github.com/Motoki0705/tennis-lab/blob/87d4fff260f2cf5e557418b98c4a77e0bf5dbcd9/outputs/explain-html/qualification/imagegen-prompt.txt)も保存しました。

親10条件との対応表はPR本文に記載し、人間の評価欄は空欄です。Geminiのtimeout解消と追加実機、上流のCSP・figure契約・deadlineの調整範囲を親へ返します。サービスは停止済み、worktreeはcleanです。