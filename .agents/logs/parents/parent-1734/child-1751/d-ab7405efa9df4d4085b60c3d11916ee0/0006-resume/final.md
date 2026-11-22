正式レビューの4件を修正し、[PR #2182](https://github.com/Motoki0705/tennis-lab/pull/2182)へ反映しました。

- branch: `codex/scene-inference-runtime`[local path]`e311c078a`
- 計画時SHAの照合、解放後の完了確定、Court外部資産の識別、例外連鎖のモデル参照解放を修正。
- 追加21ケースを含むCPU342件＋既存CLI2件、pre-commit成功。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44244540243)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44244540292)も全成功です。

cache identityへの影響と再生成の必要性をPRに明記しました。追加GPU実行は0回です。初回CIの既存注釈worker失敗は同じHEADの再実行で成功しましたが、原因は未確定です。

PRはReadyを維持しています。統合レビュー・merge・closeを親へ返します。