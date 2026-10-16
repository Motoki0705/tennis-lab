# 原要件（今回のタスク）

> src/tasks/blcs/models/discriminators/__init__.pyこのようにdiscを作ってください。入力は、[B, 128, 2]として、単純に2d refinerの出力を評価させてください。また、generatorはsrc/utils/models/componentsを使ったtransformerを作成して下さい。RoPEで位置埋め込みを入れて次元は256をベースにFFN dimは8/3倍でお願いします。層の深さは８にしてください。discは4層にして、次元は同じにします。GANのロスは現在0.002ですが、最大２に変更してください。徐々に増やしていく形です。src/tasks/blcs/configs/training/_gan.yamlこのような形でワームアップ -> 増加といったように。3d refinerも同様の変更を入れてください。blcs. plcsのディレクトリ構造や実装方法をよく参考にしてください。validatorは１回でお願いします。その後2D, 3DともにGANで学習を入れてください。不明点などありますか？

# 確定した追加指示

> 従来の開始時刻を維持：500 step待機→1,000 stepで0から2へ増加

直前にユーザーは入力のjitterと外れ値ノイズを両方0にすると明言し、「連続的なイベントの欠損のみを残す」条件を指定した。
前提: 入力は座標と欠損maskだけ、offline、全frameを1本の軌道で予測。イベントごとの選択・前後各3〜10frameの非対称欠損。3DのFlow方式は比較用に残す。
共通data/ball_refiner/single_objectを2D/3Dで共有。元データや既存checkpointは保持する。
説明済みの未変更条件: batch32, T128, 4000 updates, train/eval event選択率0.5, seed42/eval20991, 全frame SmoothL1 beta .02, G/D AdamW lr3e-4 wd .01 clip1, Gのみcosine。
FFN整数幅は既存共通部品default_ffn_dimの8/3倍・64倍数切上げ規約（256→704）。G/D4heads, RoPE64, theta10000を採用するとユーザーへ伝達済み。

# 評価対象

今回のコード・設定・通常検証。4,000 step本学習は独立評価の後に親が開始する。
コード変更と相関するUIのノイズ0対応・旧v1 checkpoint互換性も対象。旧UIタスクのvalidator指定1回とは別の新規タスク。今回の指定は1回。
元ユーザーはDataset Reviewの評価時にスクリーンショットを撮って目視確認するよう求めていたため、今回も変更したノイズ0設定とモデル選択を画面で確認する。
