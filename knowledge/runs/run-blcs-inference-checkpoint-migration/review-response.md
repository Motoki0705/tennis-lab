# PR #1097 の既存Claudeレビューへの対応

対象は[2026-10-13のレビュー](https://github.com/Motoki0705/tennis-lab/pull/1097#issuecomment-6156239935)（head `c02c20e45ab8d33e8b98d15933a4e4b8e0547ef1`）。原文を`claude-review.json`に保存した。以下は実装担当による判断・修正であり、Claudeによる修正後の承認ではない。ユーザー指定により2回目のClaudeレビューは起動しない。

| 指摘 | 判断と対応 | 検証 |
|---|---|---|
| major: 既存checkpointの公開推論・UI判定が破壊され、実checkpointテストが弱まった | 採用。既存重みを別artifactへ明示移行し、`physics_head=false`だけを設定に追加。通常推論は学習loss／metrics schemaから分離してモデルをstrict復元する。UIも同じモデル設定検証を使用。実checkpointが1件以上読み込めるlocal_dataテストを復元した。旧学習metadataの書き換え案は採らず、学習履歴を保持した。 | 131 tensor完全一致。旧・新の公開predictor出力最大差0 m。未知prefix・欠損・余分なkey・shape不一致を拒否する単体テスト、未移行版のUI拒否、移行版の実読み込みが通過。 |
| minor: 旧headline表とtrain rolloutの常時NaN | 採用。共有READMEのキー・未定義値の扱いを更新し、trainはrolloutを含まない10指標、val/testは12指標のstage別契約にした。 | train epochのflushまで通し、rollout keyが出力されない回帰テストが通過。 |
| minor: 新損失Aを標準設定で暗黙に採用している | 採用。標準は従来の位置のみ、chunkedは従来の位置＋再投影とし、新物理損失は`loss=physics`で明示する。6arm runnerは各重みを明示指定する。 | 通常設定と全6armのcompose・重み検査、configuration監査が通過。既存10runの結果・採用保留は変更していない。 |
| nit: 座標尺度・重力の重複、manifestのcwd依存 | 採用。共有正規化関数と`config.physics.gravity`を使用し、実観測bundleに重力を保存。旧bundleの再計算は明示的なgravity指定を必須にした。固定manifestを参照する一括評価controllerはユーザーの整理依頼に沿って削除し、当時のcommitで再現する。 | 保存gravityの使用・矛盾指定・旧bundleの指定漏れを検査。公開CLIの設定境界テストが通過。 |
| nit: clean targetの死んだ分岐、docstring空行 | 採用。clean target必須を維持したまま固定True分岐と冗長assert、空行を除去。 | dataset・loss回帰テストとruff/mypyが通過。 |
| nit: 選定10重みがworktree内のpathに依存 | 採用。10checkpointをworktree外へコピーし元SHAと照合。各runの現行参照と移設先receiptを更新し、元のraw receiptは履歴として保持。 | 10件・6,235,579,406 bytesのSHA一致を`checkpoint-storage.json`に記録。 |

アブレーション実装は`src/tasks/blcs/ablation/`、その公開CLIはユーザー指定の`src/tasks/ball_detection/scripts/`へ移動した。固定10runの再試行・一括評価・report controllerは削除し、実行当時のcommit・固定プロトコル・raw証拠をknowledgeに保持した。今回の移行と保存先変更に使った一度限りのscriptもこの再現bundle内に置き、常設のアプリケーション入口にはしていない。

通常検証の記録は`checks.json`と添付ログを参照する。重複するテスト範囲は合算しない。最初の実行で見つかった、変更前の既定損失を仮定するテスト、移動後の名前参照、UI fixtureとsnapshot名判定を修正し、影響範囲を再実行した。CIの成否はPRの対象headに付くcheckを正とし、ここでは将来のCI成功を先取りしていない。
