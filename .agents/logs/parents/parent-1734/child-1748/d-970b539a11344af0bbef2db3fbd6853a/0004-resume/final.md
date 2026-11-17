上限変更は完了済み、C1の全段基準値は未完です。

- [PR #2229](https://github.com/Motoki0705/tennis-lab/pull/2229)：既定上限2048。1055 frameのCPU検証で出力一致を確認。`codex/scene-frame-sampling-limit` / `06859cd87`。71テスト・pre-commit・CI成功。
- [PR #2154](https://github.com/Motoki0705/tennis-lab/pull/2154)：`codex/video-batch-baseline` / `a40d08f58`。135テスト・pre-commit・CI全14ジョブ成功。draft維持。

037は人物対応で、各sideの2候補が同率となり停止しました。停止prefixは1450.508秒、モデルloadは269.474秒（18.58%）。証拠はknowledge `000048`に保存済みです。HMR2以降・warm/cache条件は未計測です。

再開時は未commit変更なし。driver 621.24の`nvidia-smi`応答のみ確認し、CUDAは再投入していません。条件を変えず停止する指示に従い、片側人数・人物選別条件の扱いを親判断へ返します。merge・closeは未実施です。