# 続き（同じsession・branch・worktree）

投入スクリプト `outputs/trt-dino/enqueue-v4.sh` の5件を確認した。このターンで投入してよい（共有queueの worker は親側で RUNNING を確認済み。queue worker の start・serve はしない）。

- 投入したら、出力された `.job` 付きの5件のIDを `PARENT_RESULT` の queue 待ちで宣言してターンを終える。結果は parent CLI が同じsessionへ渡す。自分で待たない。
- 結果を受け取ったターンで、各ジョブの終了段階・実行層の精度・指標を照合し、knowledgeへ統合する。プラグイン版の条件2は比較用なので、nsysのみでよい（GridSample版の条件2を正式な条件2として全指標で記録する）。
- 「次のターンで投入する」という前回の親指示は、投入前に止まれという意味ではない。以後、投入準備ができたら同じターンで投入して宣言する。
