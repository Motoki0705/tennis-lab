RT-DETRv4 L/X の診断2件を検証し、`codex/parent-2492-issue-2487-rtdetrv4` の `fef762822f6f8f7e62ae0ec5d0fc5b80eee15447` を push しました。 [#2487 の検証記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7162619357)

代表6 clip・6,844画像の結果です。parity は全条件で「PT32未計測」です。

| モデル | Backend | Recall@0.50 | Recall@0.75 | 比較での扱い |
|---|---|---:|---:|---|
| L | PT16 | 98.12% | 45.57% | 比較対象 |
| L | TRT16 | 0.00% | 0.00% | 精度低下で除外 |
| X | PT16 | 97.97% | 44.66% | 比較対象 |
| X | TRT16 | 0.03% | 0.00% | 精度低下で除外 |

同じ4画像を使った診断では、L/X とも batch 1・2 で PT16 と ONNX Runtime CPU が IoU 0.50 で16件中16件、TRT16 が0件でした。保存した生出力から全12ケースの採点結果を再現しました。大きな差は共通の後処理へ渡す前の TensorRT logits と BBox に現れており、原因となる演算は未特定です。両 TRT16 plan の速度は検証記録に保存しました。 [診断・再現手順](https://github.com/Motoki0705/tennis-lab/blob/fef762822f6f8f7e62ae0ec5d0fc5b80eee15447/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4.md) · [job ID・生出力の場所・SHA-256](https://github.com/Motoki0705/tennis-lab/blob/fef762822f6f8f7e62ae0ec5d0fc5b80eee15447/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4-results.json)

pytest 82件（10.97秒）、pre-commit、knowledge 検証が通過しました。補助ブランチのCIは起動済みで、結果は未確認です。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45615388107)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}