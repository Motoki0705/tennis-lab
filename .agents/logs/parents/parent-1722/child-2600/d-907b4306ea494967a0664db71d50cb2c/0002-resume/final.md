共有データ・checkpoint・出力先のシンボリックリンクを解決してから検証するよう修正し、[PR #2613](https://github.com/Motoki0705/tennis-lab/pull/2613) に push しました。HEAD は `976a71056e562a19f6a646c8004c4f081282246a` です。

同じ失敗を再現するよう統合テストを拡張し、修正後は関連12件と pre-commit が通過しました。実データの69本の選手IDと独立ラベル6本のハッシュ・フレーム軸も照合しました。

固定入力でCPU前処理を再起動しました。GPU queue への投入は前処理の結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T070346Z_236ec63d5ed9_meiji-v3-motion-inputs-r2"]}}