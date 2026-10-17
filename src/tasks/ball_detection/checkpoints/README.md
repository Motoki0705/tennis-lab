# Ball checkpoint の入力契約

新しい `MDDPretrainConfig`、その派生設定、YAML は `input_channels` を必須とする。
保存時にもモデル設定へ明示する。チャンネル数だけで入力の意味を決めず、保存した
`recipe.input_contract` 全体とモデルの変換を厳密に照合する。

既存の DPT pretraining v1/v2 と query posttraining v1 には `input_channels` のない
保存形式がある。これらだけは `configuration.checkpoint_model_config` で、保存済みの
`rgb_uint8_native_mdd.v1` 契約が係数を含む全フィールドで既存 `RGBToMDD` と一致する
場合に限り2chと識別する。checkpoint schemaだけ、重みの形状だけ、ファイル名だけ
では識別しない。契約の欠落・改変・未知schema・5ch契約での指定欠落は拒否する。
DPT v1 の residual/dense3d 構成もschemaの定義に限定して復元する。

この処理はメモリ上の設定だけを作り、checkpoint・run設定・annotationの固定specを
書き換えない。注釈推論、座標評価、encoder初期化、学習再開は同じ復元関数を通す。
再開時のrecipe比較は識別済みモデル設定のみを正規化し、元のrecipeとmodel_configの
不一致、データ・実装hash・学習条件の変更は引き続き拒否する。過去runの厳密再開は
記録されたコード版で行う。新しいコード版での重み初期化とは区別する。

`mdd_coordinates.v4` の圧縮比較モデルは `MDDPoseConfig` と保存済み入力契約を使い、
このCNN設定の移行対象にはしない。

`mdd_hybrid_training.v1` は新規5chモデル専用で、`input_channels=5` の明示が必須。
評価・推論・再開・継続学習は同じ `restore_hybrid_model` で検証する。
queryからの初期化も入力契約を照合し、履歴2chモデルから5chへの転送は拒否する。
