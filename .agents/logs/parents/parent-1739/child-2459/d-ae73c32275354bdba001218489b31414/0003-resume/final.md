FP32キャッシュ互換を修正し、同じブランチの [PR #2461](https://github.com/Motoki0705/tennis-lab/pull/2461) にpushしました。比較機能のPython差分7ファイルが記録したハッシュとすべて一致した場合に、従来の互換判定を適用します。

CIで失敗した2件を含む50件とpre-commitが通りました。旧FP32キャッシュを `execute`・`load` の両方で再利用でき、FP32・TensorRTとコート向きの組み合わせ4条件で、比較機能追加前の実行identityと一致しました。比較ページの16本も採用中のシーンと再照合済みです。

#2446 の積み直しでは `source_identity.py` の競合解消と、合成後のハッシュ照合が残ります。

ブランチ `codex/parent-1739-issue-2459-primary`  
新HEAD `a80c7a8c3e4bd31eb192f25a77b34f7bad60cb6a` のCIは結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2461,"head":"a80c7a8c3e4bd31eb192f25a77b34f7bad60cb6a","resume_on_success":false}}