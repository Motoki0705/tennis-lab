# Camera-local view association

PLCS/BLCSの2D検出を、カメラ校正と三角測量の前に対応付けるモデルです。
`train_association` entry pointで学習します。従来の3D track-queryモデルとは
別のチェックポイント契約を持ち、再学習が必須です。Courtの入力契約は
[共有正本](generate_dataset/README.md)を参照してください。

入力は`object_uv (B,V,T,P,J,2)`、`object_vis (B,V,T,P,J)`、
`court_kp (B,V,T,14,2)`、`court_vis (B,V,T,14)`、`padding_mask (B,V,T)`、
`reference_view_index (B)`。PLCSはJ=17、BLCSはJ=1です。side、カメラ姿勢、
3D座標、GT identityはモデル入力に含みません。

各stageはobject tokenをmHCでP→1に圧縮し、viewごとのQを時間列の先頭へ
追加します。時間attentionは`(B*V,T+1,D)`、空間attentionは`(B*T,V,D)`です。
Qは空間attentionへ入りません。Qの位置は-1とし、時間attentionは全域MHAだけを
使います。既存CSWAの局所windowへQを通常フレームとして入れる解釈はしません。
空間RoPEの3軸は時刻、view、reference selectorです。mHCで元のobject streamへ
書き戻し、最終Qをside head、object tokenをidentity headへ渡します。

出力は`side_logits (B,V)`と`object_id_logits (B,V,T,P,K+1)`です。
Kはclip内の総identity数の上限で、同時存在数Pとは独立です。最後のclassは
false positiveです。GTは2D trackerの完了後に物理instance provenanceから
教師専用fieldへコピーします。全view・全時刻に対する1回のHungarian matching
でclass番号の任意性を除き、観測があるobjectだけを分類します。参照viewの
sideは定義上falseなのでside損失・正解率から除きます。

各taskの`PLCSAssociationPredictor` / `BLCSAssociationPredictor`は同一view/frame内のID重複を解消し、欠測・非対象を-1、
参照sideをfalseにします。ID番号は1回のモデル入力区間内でのみ有効です。同じraw
観測を共有する重複window用の補助関数
[`stitch_overlap_ids`](../../tennis_scene/pipeline/utilts/association_state.py)は、
現行predictor/pipelineには未接続です。predictor自身は入力を自動分割しません。
重複区間に現れない新しいIDは呼び出し側が新規scene IDを割り当てます。

下流の`src.tennis_scene.pipeline.components.view_association.ViewAssociationModule`
はstable camera IDからreferenceを指定し、結果を返します。結果の
`geometry_context()`で推論済みsideによるカメラ校正、`group_observations()`で
同一IDの観測をまとめます。3Dの三角測量・補正は下流の責任です。

```bash
# ローカルGPUではtraining-queue経由で実行する。
.venv/bin/python -m src.tasks.plcs.scripts.train_association
.venv/bin/python -m src.tasks.blcs.scripts.train_association
```

チェックポイントは`run.output_dir/logs/version_*/checkpoints/`、loss/side・ID正解率は
`association_metrics.jsonl`とTensorBoardに保存します。`run.fast_dev_run=true`
と`training.warmup_steps=0`で1batchの実データスモーク検証ができます。
全欠測windowも通常のpadding/visibility契約で扱い、3D損失は計算しません。

実モデルの設定は各taskの`configs/model/view_association.yaml`だけを正本とします。
旧`association.model`と3D tracking model設定の併存は廃止しました。
汎用stageは`utils/models/components/view_query.py`、Court encoderとside/ID headは
`tasks/base/models/view_association.py`へ配置します。入力と教師は各taskのassociation
Dataset、入出力はmodel_io、学習は既存task runner/compositionで構成します。
時間attentionは全stage Global MHA、mHCは保持します。CSWA/CUDA拡張は使いません。
データはcamera-local 2D観測とside/ID教師のみを読み、3D target packingを行いません。
共通optimizer/scheduler/compileを有効にし、新しい契約v2として両taskを新規60epoch学習します。
