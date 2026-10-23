# decision — ユーザーの判断

ユーザーの一般的な判断・好み。マージ可否の判断基準は [review](../review/index.md) に置く。1行に1エントリを書き、リンク文字列はエントリの `title` と一致させる。形式は [memory skill](../../skills/memory/SKILL.md#形式) を参照。

- [判断を仰ぐときは、既に答えが出ている事項を聞き直さず、各選択肢を説明付きで示す](ask-only-open-decisions.md)
- [test推論をbackfillできない古いckptは、結果がknowledgeノードにあれば削除してよい（ユーザー判断）](ckpt-force-delete-policy.md)
- [学習などの長時間ジョブは、定期的な再開ではなくプロセスの監視で待つ](wait-for-long-jobs-by-event.md) (claude)
- [注釈workerのeffortはmaxに固定する](annotation-worker-effort-max.md)
- [テスト用の値を本番planに使わない](keep-test-values-out-of-production-plans.md)
- [公開向け文書は英語版を既定にして日本語・簡体字中国語版を別ファイルに分け、ハッシュ値は書かない](public-docs-english-default.md)
