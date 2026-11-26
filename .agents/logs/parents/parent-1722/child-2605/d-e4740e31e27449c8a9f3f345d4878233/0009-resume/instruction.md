# PR #2610 正式レビュー（round 1）の major 修正

対象HEAD `555a246aabb7201959e3bb28cc6f40079e62b5c3`。Sol の正式レビューで major 3件が出た。親は3件とも採用する。各指摘に対応する修正 commit と、失敗を再現してから通る回帰テストを返す。

## 1. クリーンインストールが事前診断で必ず停止する

`scripts/prepare_clean_install.py:149` と導入 README の3言語版は、`event_detection.method` を指定せずに `scripts.preflight` を実行する。事前診断は既定の `null` で `PipelineRuntimeConfig.from_config` を呼ぶため、`pipeline_configuration` エラーで終了値1になる。自動導入と手動手順の両方を、後続の再構成コマンドと同じ明示設定で通す。既存 CLI テストに、正常時に終了値0になるケースを足す。

## 2. 文書どおりの TensorRT モデル export が起動できない

`pipeline/README.md:406` の `tensorrt export vitpose "$bundle"` は、`scripts/tensorrt.py:215,251–254` で空の override から完全な pipeline 設定を読む。`bind_inputs=False` でも method が必須なので、5モデルすべての export が設定エラーになる。

方針: event_detection を実行しない経路（モデル export など）に、イベント方式の選択を要求しない。ただし既定の方式を黙って補うことはしない（例えば `rule` を内部で入れて通すのは不可）。event_detection の設定検証を、その component を実行する経路だけに結び付ける。モデル本体を起動せずに、CLI の設定経路をテストで確かめる。

## 3. 通常の annotation 方式が別映像に正解ラベルを付けられる

元注釈69件のうち `clip_040` を含む20件では、`ball_events/annotation.py:91–94` の正規表現で映像 SHA-256 が1件も取れない。そのため通常の pipeline は、クリップID・FPS・フレーム数が同じ別映像でも、映像を照合せずにラベルを採用できる。注釈仕様が求める固定 v2 manifest との映像照合は、いまはベンチマーク生成側にしかない。

通常の annotation 方式にも、固定した映像ハッシュを入力として束縛する。照合できない場合は拒否する。フレーム軸が同じで映像ハッシュだけが違う場合に拒否するテストを足す。注釈フォルダはユーザーの成果物なので、読むだけにし、ハッシュを書き足さない。

## 全体の確認

- repo 内で `PipelineRuntimeConfig.from_config`（またはそれを経由して完全設定を読むもの）を呼ぶ入口を列挙する。それぞれ「event_detection を実行する／しない」に分け、1・2と同じ問題が残っていないかを確かめる。結果を PR 本文に短く書く。
- 修正で multiview-v3 の component の識別（artifact_id・execution_key など）が変わるなら、最終HEADで69クリップを再生成・再照合する。そのうえで、イベント配列と coverage が前回の配布版（manifest SHA-256 `40efd634…`）とバイト一致するかを報告する。後続の #2600 がこのデータを入力にしているため。識別が変わらないなら、その根拠を書く。
- PR 本文の実行証拠を最終HEADで更新し、push 後は CI を `PARENT_RESULT` の wait で宣言して終了する。
