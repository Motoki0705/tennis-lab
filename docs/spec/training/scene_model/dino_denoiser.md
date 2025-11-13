# DinoDenoiser 仕様

`src/training/scene_model/dino_denoiser.py`: DINO 風のノイズを GT に付与する簡易デノイザ。

## 目的
- 各フレームの GT ボックス/ID に摂動を加え、学習時のノイズ付きクエリを生成する。
- 現状 `HeadAdapter.compute_loss` では未使用（将来的な拡張フック）。

## 入出力と形状
- 入力: `targets: Sequence[Sequence[TargetFrame]]`（バッチ×フレーム）
- 出力: `DenoiserState`
  - `boxes:  Float[M, Qn, 4]`（`[cx,cy,w,h]`）
  - `labels: Long[M, Qn]`（-1 でパディング）
  - `pad_size: int (=Qn)`
  - ここで `M` はバッチ内の有効フレーム総数（`~padding_mask`）と一致させる想定。

## 擬似コード

```
__init__(cfg):
  Qn   = cfg.num_noisy_queries (>=0)
  s_box= cfg.box_noise_scale
  p_id = cfg.label_noise_scale

make_noise(targets):
  if Qn <= 0:
    return zeros(shape=[0,0,4], [0,0])

  boxes_list, labels_list = [], []
  for sample in targets:       # 各サンプル
    for frame in sample:       # 各フレーム（パディング含む）
      cxcywh, ids = frame_noise(frame)
      boxes_list.append(cxcywh)      # [Qn,4]
      labels_list.append(ids)        # [Qn]

  if not boxes_list:
    return zeros(shape=[0,Qn,4], [0,Qn])

  return DenoiserState(stack(boxes_list), stack(labels_list), pad_size=Qn)

frame_noise(target):
  boxes = concat([target.center, target.size], dim=-1)  # [N,4]
  take = min(N, Qn)
  padded = zeros([Qn,4]); padded[:take] = boxes[:take] + s_box * randn([take,4])
  labels = full([Qn], -1)
  if take>0 and target.track_ids:
    base = target.track_ids[:take]
    with prob p_id, 一部をランダム ID に置換
    labels[:take] = base
  return padded, labels
```

## 設定キー
- `training.denoiser.num_noisy_queries: int`
- `training.denoiser.box_noise_scale: float`
- `training.denoiser.label_noise_scale: float`（`label_noise_prob` 相当）

