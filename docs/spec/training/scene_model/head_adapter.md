# HeadAdapter（損失アダプタ）仕様

`src/training/scene_model/head_adapter.py`: SceneModel の出力を DINO 由来の目的関数へ接続する。

## 目的
- 各フレームの予測と GT の対応付け（ハンガリアンマッチング）。
- 分類（presence）、存在確率、BBox L1、GIoU の損失を計算し、重み付きで合算。

## 入出力と形状
- 入力: `outputs: Mapping[str, Tensor]`, `targets: list[list[TargetFrame]]`, `padding_mask: Bool[B,T]`, `dn_state`（現状未使用）
- 正規化済みの有効フレームのみを使用:
  - `flat_boxes:   Float[Nf, Q, 4]`（`[cx,cy,w,h]`）
  - `flat_logits:  Float[Nf, Q, C]`
  - `flat_exist:   Float[Nf, Q]`（ロジット）
  - `target_list:  list[TargetFrame]`（長さ Nf）
  - ここで `Nf = sum_{b,t} 1{~padding_mask[b,t]}`

## 損失の定義
- マッチング: `HungarianMatcher(cost_class, cost_bbox, cost_giou)`（third_party DINO が利用可能ならそれを使用。不可なら `_GreedyMatcher` で簡易対応）
- 分類: `CrossEntropy(logits, targets)`
  - ターゲットは、マッチしたクエリ=クラス1、それ以外=クラス0 の二値（デフォルト `num_classes=2`）
- 存在: `BCEWithLogits(exist_logits, exist_target)`
  - マッチしたクエリ=1.0、それ以外=0.0
- BBox L1: `L1(pred_cxcywh, tgt_cxcywh)`（合計をフレーム内 GT 数で正規化）
- GIoU: `mean(1 - diag(GIoU(pred_xyxy, tgt_xyxy)))`（同正規化）

合計損失:

```
total = w_cls * cls + w_bbox * bbox + w_giou * giou + w_exist * exist
```

返り値:

```
{
  total, cls, bbox, giou, exist,
  num_matches: float tensor,
  num_targets: float tensor,
}
```

## 擬似コード

```
def compute_loss(outputs, targets, padding_mask, dn_state=None):
  pred_boxes = extract_pred_boxes(outputs)         # [B,T,Q,4]
  cls_logits = outputs['cls_logits' or 'role_logits']
  exist      = outputs['exist_conf'].squeeze(-1)

  # 有効フレームに限定
  valid = ~padding_mask
  flat_boxes  = pred_boxes.view(B*T,Q,4)[valid]
  flat_logits = cls_logits.view(B*T,Q,C)[valid]
  flat_exist  = exist.view(B*T,Q)[valid]
  tgt_list    = list(iter_valid_targets(targets, padding_mask))
  if not tgt_list:
    return zeros

  match = matcher({'pred_logits': flat_logits, 'pred_boxes': flat_boxes},
                  [target_to_dict(t) for t in tgt_list])

  cls_loss   = classification_loss(flat_logits, match)
  exist_loss = exist_loss(flat_exist, match)
  bbox_loss, giou_loss = bbox_losses(flat_boxes, match, tgt_list)
  total = w_cls*cls + w_bbox*bbox + w_giou*giou + w_exist*exist

  return {...}
```

## 出力抽出ルール
- BBoxHead の場合: `bbox_center` と `bbox_size` を連結。
- PredictionHead の場合（BBoxHead でない）: `ball_xyz[..., :2]` を中心、`|player_xyz[..., :2]|` をサイズとして代用（常に正）。

## 設定キー（例）
- `model.head.matcher.{cost_class,cost_bbox,cost_giou}`
- `model.head.loss.{cls_weight,bbox_weight,giou_weight,exist_weight}`
- `model.head.type: 'bbox' | 'scene'`（ビルダー側で選択）

