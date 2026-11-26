PR #2610の統合HEAD `2999c0e743805fb2145ae6e03121f81b8c98df0c` で再生成した `multiview-v3` を配布先に反映しました。新しいmanifestのSHA-256は `790f05016e31ff4eb3fa8fc7d681d1a22ff0540928672e2a274a64543cd2a28a` です。 [配布manifest](https://github.com/Motoki0705/tennis-lab/blob/2999c0e743805fb2145ae6e03121f81b8c98df0c/data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3/manifest.json)

- #2600の学習版と、採用 `scene.npz` 全69件をバイト比較しました。イベント配列・coverage・v2由来の配列を含む全2,553メンバーと、train／val／testの分割が一致しました。
- 切り替え直前の配布版とも、イベント配列・coverageの全552個がバイト一致しました。直前版は `.multiview-v3-cba442a2d` に保管しました。

PR本文に実行証拠と、正式レビュー3件それぞれの修正コミット・回帰テストの対応を記録しました。統合HEADのPython CIは成功済みです。 [PR #2610](https://github.com/Motoki0705/tennis-lab/pull/2610)・[CI結果](https://github.com/Motoki0705/tennis-lab/actions/runs/45791298435)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}