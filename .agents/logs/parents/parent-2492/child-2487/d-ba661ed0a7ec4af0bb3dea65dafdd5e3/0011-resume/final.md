[PR #2499](https://github.com/Motoki0705/tennis-lab/pull/2499) に `codex/parent-2492-issue-2487-primary` の HEAD `f5f1a5f2b1bcfd69b55ebcd217c73426b6393984` をpushし、PR本文の実行証拠を更新しました。#2501と#2573を含むmainを取り込み、FastHMR・単眼PLCSとECPoseの定義を統合しました。GitHub上でマージ可能です。

ECPose本体・BBox・COCO17は、同じ実行設定でcomponentが生成した成果だけを再利用します。公式重みのSHA-256を計画時と公開用の識別情報を作る時点で照合し、PyTorchモデルはCPUで復元してから指定デバイスへ移します。

標準FP32キャッシュの再利用と追加回帰6件を含む関連テスト98件、境界検査18件、比較ツール47件が通過しました。pre-commitとknowledge検査も通過し、比較表44行とPareto図2枚は承認済みの成果と一致しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"f5f1a5f2b1bcfd69b55ebcd217c73426b6393984","resume_on_success":false}}