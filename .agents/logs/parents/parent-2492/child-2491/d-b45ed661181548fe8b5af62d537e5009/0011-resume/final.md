[PR #2501](https://github.com/Motoki0705/tennis-lab/pull/2501) の正式レビュー3件を修正し、同じブランチに `9c48c47a73d9472b159c18cf00dc95962e212246` をpushしました。PR本文の `Closes #2558` を維持しています。

1. FastHMRの `load` にコンポーネント生成の出自を要求し、PT16設定でのPT32インポートを拒否します。公開時も出自と精度を含む実行設定を照合します。
2. 公式資産8点のSHA-256を、計画と公開用の構成を組む時点で照合します。復元リクエストが空でも、不一致は保存前に検出します。
3. VAEをCPUに復元してから、モデルを指定デバイスへ移します。CUDAを隠した状態で公式VAEの297テンソルをCPUに復元できました。`cuda:1` への転送指定はGPUなしのテストで確認しました。

新規5ケースは修正前にすべて失敗し、修正後に通過しました。関連pytestは標準FP32キャッシュ回帰を含む130件が成功し、pre-commitも通過しました。新HEADのCI結果を待ちます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2501,"head":"9c48c47a73d9472b159c18cf00dc95962e212246","resume_on_success":false}}