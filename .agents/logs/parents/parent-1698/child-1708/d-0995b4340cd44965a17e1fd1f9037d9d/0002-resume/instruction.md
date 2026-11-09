## 統合レビューの指摘への対応（PR #1763）

統合レビューで major が1件見つかった。同じbranch・worktreeで修正し、PRへpushしてCIの成功を確認するところまで行う。

### 指摘 pr-1763-1（`src/tasks/ball_detection/annotation/video/agent/services.py` 171行付近）

同じattemptで `video_handoff` → `video_edit` → 同じ `video_handoff` を実行すると、要求のdigestが変わらない。そのため、編集前の remaining・summary・revision が返り続ける。`video_rows` も、同じframe指定では修正前の行を返す。`scripts/client.py` も既存のreplyを即座に読むので、サービス側で再計算するだけでは直らない。結果として、処理済みのunknownが未処理として再提示され、返されたrevisionで次の編集をするとCAS競合になる。進捗の確認と自己修正が妨げられる。

### 直してほしいこと

- 読取要求（handoff、rows など）でも、新しい読取と同じ要求の再送を区別する。例: 読取要求に、現在のrevisionや呼出しごとの識別子を含める。編集の後の読取では、必ず最新revisionの内容が返るようにする。再送の冪等性（同じ要求には同じ応答）は保つ。
- 回帰テストを追加する: 同じattempt内で、読取 → 編集 → 同じ範囲の再読込をすると、最新のrevision・remaining・summary・行が返り、そのrevisionで次の編集がCAS競合にならないこと。修正前のコードでこのテストが失敗することも確認する。
- ほかの読取系の道具にも同じ問題がないか確認し、あれば同じ方針で直す。

### 終了条件

修正のpush、PR本文の検証欄の更新、対象HEADのCI（`Python` を含む）成功の確認までを1回の作業単位とし、目安は2時間以内。最終報告には、修正内容、追加テストの結果、新しいHEADを書く。
