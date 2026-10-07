# 評価の扱い

指定1回、試行1回、完了1回。原判定CHANGES_REQUIREDはresult.mdに保持。

R1/P2を採用。新設したnoise-jitter/noise-outlierを既存changedInput()へ接続した。
親の通常検証で、CPU推論後に各欄だけを編集すると、previewが新値で発生し、古い予測と誤差が消え、input hashが変化することを確認。
続く再推論のinput hashがpreviewと一致し、両次元の予測が再表示されることを確認。JavaScript pageerrorは0。node --checkも成功。
スクリーンショットnoise-jitter-cleared.pngをview_imageで目視し、GT/入力のみの表示と誤差欄のクリアを確認。
証拠: ../rope-gan-20261006/noise-control-regression.json と2枚の*-cleared.png。

UI修正後にvalidatorの再評価は行っていない。本学習はこのUI修正と通常検証の後に開始した。その後の変更は以下に記す。


## 本学習後の通常検証とCI対応

2D/3Dとも4,000更新を完了し、両方でbest step=4000、GAN係数2を選択した。親が新しい両重みを選んだ比較画面を撮影・目視し、保存済みCPU予測とその場でのCPU再推論が配列で完全一致することを確認した。座標曲線はGTに近いが、欠損付近の速度に振動が残る。trained-browser-summary.jsonに対象重み・入力hash・誤差・pageerror 0を記録した。

最初のPR CIは6,375 passed、86 skipped、1 failed。新GeneratorのFFN選択が共通componentsの設定規約に未接続だった。親が設定からffn_typeを渡すように修正し、共通利用者一覧へ追加した。checkpointはFFN明示のv3にし、既存v2は定義どおりSwiGLUとして明示復元する。v1/v2/v3互換性を含む対象54テストとruff/mypyのcommit hookが成功した。今回の実際の2D/3D best重みで修正前後のCPU出力はbit単位一致した。学習run自体はc5cee39bで実行され、その記録・設定・重みは更新していない。

これらは親の修正と通常検証であり、validatorによる再評価ではない。指定1回・試行1回・完了1回を維持する。


CIの追加設定監査で、FFN設定にPython側の既定値を置いたことがcomposition-owned defaults規約に違反すると判明した。ffn_typeを必須フィールドにし、既定値はYAMLだけに保持するよう親が修正した。v2互換経路は形式に定義されたSwiGLUを明示するため、重み・演算は不変であり、実際の2D/3D best重みのCPU出力も元の学習版とbit一致した。この変更もvalidator再評価の対象外である。

この既定値修正後、Refiner・FFN選択・共通設定監査・関連consumerを合わせた通常検証で454 passed / 1 skipped。ruff/mypyのcommit hookも実施する。
