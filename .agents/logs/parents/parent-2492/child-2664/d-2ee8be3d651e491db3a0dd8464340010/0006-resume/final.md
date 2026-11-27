型付き設定への引数を明示するよう修正しました。コミットは `46c25d1b499f3f6367bcae8b6fbd758643ce3b3f` です。

出力先テストで失敗した20件は共有ストレージの配置に起因し、同じ HEAD の隔離環境では出力先テスト116件がすべて合格しました。設定・既定経路のキャッシュ互換・TensorRT 識別の40件、公式 MMPose との CPU 照合、両モデルの重みの SHA-256 照合も合格しました。

設定一覧を含む残りのテストと設定監査を再投入しました。結果確認後に push し、PR を作成します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261127T003300Z_3ec0e6d1773a_i2664-full-config-final"]}}