[PR #2200](https://github.com/Motoki0705/tennis-lab/pull/2200)を更新しました。**受入未達で、merge保留です。**

- branch: `codex/court-acceptance-postprocess`
- HEAD: `d909f9a79`
- meiji4：誤採用34件、正しいHの採用59/181＝32.6%。cam3採用0件。
- synthetic：前回の喪失58件中35件を回復しましたが、旧処理比38件の喪失が残りました。
- 最終検証：144 passed／1 skipped、pre-commit成功。[Pythonを含むCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43956279470)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43956279546)成功。未変更のannotationテストの一時失敗は、同一HEADで再実行しました。validatorは未指定・0回です。

全件比較と1,449枚の画像を[knowledge 74](https://github.com/Motoki0705/tennis-lab/blob/d909f9a79/knowledge/nodes/court_detection/000074-run-court-semantic-support-20261117.md)に記録しました。ネットと塗線の意味対応、候補順位、不確かさによる拒否の再設計が親の判断点です。参照・閾値は変更せず、merge・issue closeも行っていません。正式なCourt CUDA確認は未実施です。

初回pytestでCUDA unit 1件をqueueなしで誤実行しました。制約違反を記録・訂正し、以後はCUDAを無効化しました。新しいモデル推論・raw取得・infra操作はありません。