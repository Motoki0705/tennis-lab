外部ボール取り込みCLIを設定監査に登録し、入力・出力パスを公開前に共通のパス契約で検証するよう修正しました。ソースハッシュの起点には `PROJECT_ROOT` を使います。

CIで失敗した2件を含むローカルテスト413件、pre-commit、knowledge検査が通りました。修正後のコードで公開済み69クリップを読み戻し、クリップ別集計と参照・成果物のSHA-256が前回の記録と一致しました。

[PR #2462](https://github.com/Motoki0705/tennis-lab/pull/2462) を更新しました。新HEADは `c839934f41311027557d1b4e3ac69c6974478002` です。CI待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2462,"head":"c839934f41311027557d1b4e3ac69c6974478002","resume_on_success":false}}