# PR #2182: 最新mainの取り込み（親 #1734 から #1751 へ）

第4回のmajor 4件への対応（`caa0a01a3`）を確認しました。人間の指示で、ここからは再レビューを行いません。親が統合後のCIを確認して、そのままmergeします。

その後、mainにPR #2240（人物ステージの精度選択とDINO拡張のpin）がmergeされました。#2240は、このPRと同じ `definition.py`・`orchestrator.py`・人物系componentを変更しています。GitHub上で衝突が出ていなくても、意味の上で食い違う可能性があります。

1. `origin/main` を `codex/scene-inference-runtime` にmergeする。衝突があれば両方の意味を保って解く。
2. 統合後に、次の点を確かめる。
   - #2240の精度設定（fp16/fp32）とDINO拡張のpinが、このPRのModelSpec・cache identity・ランタイム（モデルの貸し出し・解放）の上で正しく働くこと。
   - このPRのA〜Eの横断確認（特に、注入経路と失敗の伝播）が、#2240で追加された経路でも成り立つこと。
   - 足りないテストがあれば追加する。
3. `tests/unit/tennis_scene` 全体と、関連するCPUテスト、pre-commitを実行する。GPUは使わない。
4. pushしてPR CIを起動し、対象HEADで `Python` を含む全ジョブが成功するまで見届ける。
5. PR本文に、取り込みの内容と確認の結果を追記する。新しい機能や別のリファクタリングは入れない。

CUDAのエラーやハングが起きたら、再試行せずに止めて報告すること。最後に、HEADのSHA・CIのURL・確認の結果を簡潔に報告すること。
