# Component 5: BLCS model, data, mask, and configuration integration

## 1. Responsibility

このコンポーネントは Components 1–4 を `BLCSTrackQueryModel` へ統合し、BLCS tracking data/I/O/configuration を fixed candidate width と hybrid temporal schedule に更新する。shared primitive の内部実装や CUDA kernel は変更しない。

統合後の stage は必ず次の順序を持つ。

```text
fixed-width camera tokens [B,V,T,Q,D]
  1. mHC pre Q->1
  2. object temporal: CSWA or Global over T
  3. mHC post 1->Q + constrained residual mixing
  4. spatial MHA over Q + VQ per frame
  5. query temporal: same CSWA or Global mode over T
```

## 2. Fixed observation-candidate contract

model input は次へ固定する。

```text
ball_uv:        [B,V,T,Q,2]
ball_visible:   [B,V,T,Q]
candidate_mask: [B,V,T,Q]
court_kp:       [B,V,T,14,2]
court_vis:      [B,V,T,14]
frame_mask:     [B,T]
view_mask:      [B,V]
```

`candidate_mask` は「その camera/time/candidate slot に実在する observation candidate が割り当てられている」ことを表す。visibilityとは異なる。

```text
ball_visible => candidate_mask
candidate_mask => frame_mask and view_mask
```

candidate が実在するが camera から見えない場合は

```text
candidate_mask = true
ball_visible   = false
```

である。これにより `mask_invisible_observations=false` のときも、padding slot と invisible-token slotを区別できる。

`P` は runtime tensorから決めず、常に `Q=model.num_queries` とする。`ball_uv.shape[3] != Q` は model-I/O boundary で reject する。

## 3. Dataset packing

scene の physical track 数は clip 全体では `Q` を超えてよいが、同一frameの concurrent candidates は `Q` 以下でなければならない。候補を黙って先頭 `Q` 件へ切り捨てない。

Dataset は次の二つの lifecycle assignment を概念的に分離する。

1. target assignment: 既存の persistent query target `[T,Q]` を作る。
2. observation assignment: camera observationsを fixed candidate streams `[V,T,Q]` へ詰める。

observation assignment は全viewで共通にし、同じ physical object が同一frameでcameraごとに別candidate indexへ移らないようにする。また object temporal path が意味を持つよう、physical lifecycle中は可能な限り同じ observation streamを維持し、lifecycle終了後のみreuseする。

trainingで slot randomization を使う場合、observation assignment と target assignment の permutation は独立にsampleし、candidate indexがtarget query indexを直接表す shortcut を作らない。evaluationでは再現可能な deterministic assignment を使う。Q=1など独立 permutationが意味を持たない場合は例外とする。

packing result:

```text
packed_uv:          [V,T,Q,2]
packed_visible:     [V,T,Q]
packed_candidate:   [V,T,Q]
packed_gt_index:    [V,T,Q]
```

padding slotは `uv=0`, `visible=false`, `candidate_mask=false`, `gt_index=-1` とする。候補がpresentだがinvisibleな場合、`candidate_mask=true`, `visible=false` とし、debug/evidence用 `gt_index` はphysical IDを保持してよい。

collate時にcandidate axisを可変padding対象にしない。各sampleがdataset境界ですでにexact `Q` であることをassertする。

## 4. Observation encoder contract

既存 observation encoder の public return を

```text
camera_tokens:      [B,T,V,Q,D]
camera_state_valid: [B,T,V,Q]
```

として維持または明示する。model は直後に canonical internal layoutへ変換する。

```text
camera_tokens = permute -> [B,V,T,Q,D]
camera_state_valid = permute -> [B,V,T,Q]
```

state validityは boundary で次のように定義する。

```text
context_valid = view_mask[:, :, None, None]
                and frame_mask[:, None, :, None]

candidate_context_valid = context_valid and candidate_mask

camera_state_valid = candidate_context_valid and ball_visible
                     if mask_invisible_observations
                     else candidate_context_valid
```

point-attention fusionも `candidate_mask` を使う。padding candidateを invisible tokenとして取り込まない。

## 5. Boundary-prepared masks

`prepare_tracking_attention_masks()` は次を返す厳密な contractへ更新する。

```text
camera_state_valid:            [B,V,T,Q]
spatial_attention_mask:        [B*T,Q+VQ,Q+VQ]
object_temporal_state_valid:   [B*V,T]
object_temporal_attention_mask:[B*V,T,T]
query_temporal_state_valid:    [B*Q,T]
query_temporal_attention_mask: [B*Q,T,T]
point_attention_mask:          point-fusion contract
```

定義:

```text
object_temporal_state_valid[b,v,t]
  = any_q camera_state_valid[b,v,t,q]

query_temporal_state_valid[b,q,t]
  = frame_mask[b,t]
```

spatial valid sequenceは各 `(b,t)` で

```text
concat(
  frame_mask repeated Q times,
  camera_state_valid over V,Q
)
```

とする。Global stageはdense keep maskを使い、CSWA stageはraw state-valid maskを使う。model内でmaskの意味を推測・再構成しない。

## 6. Stage construction

新しい task-local moduleを推奨する。

```text
src/tasks/blcs/models/components/track_query_stage.py
```

```python
class BLCSTrackQueryStage(nn.Module):
    def __init__(
        self,
        *,
        stage_index: int,
        mhc: ManifoldConstrainedHyperConnection,
        object_temporal_block: TransformerBlock,
        spatial_block: TransformerBlock,
        query_temporal_block: TransformerBlock,
        hidden_dim: int,
        num_queries: int,
    ) -> None:
        ...
```

stage modeはconstructor時に固定する。

```text
is_global = stage_index % 4 == 3
```

runtime flagでCSWA/Globalを切り替えない。`num_stages` は正の4の倍数を要求し、未完の3:1 cycleを許可しない。

object/query temporal blockは同じmodeだがparameterは共有しない。mHCもstageごとに独立parameterを持つ。spatial blockは常にglobal MHAである。

## 7. Stage forward

canonical input:

```text
C_s: [B,V,T,Q,D]
S_s: [B,T,Q,D]
```

### 7.1 Object temporal path

```python
projected, mhc_state = self.mhc.pre(
    C_s,
    camera_state_valid,
)
# projected: [B,V,T,1,D]

object_values = projected.squeeze(-2).reshape(B * V, T, D)

if self.is_global:
    object_update = self.object_temporal_block.forward_update(
        object_values,
        freqs_cis=time_freqs,
        attn_mask=object_temporal_attention_mask,
    )
else:
    object_update = self.object_temporal_block.forward_update(
        object_values,
        freqs_cis=time_freqs,
        state_valid=object_temporal_state_valid,
    )

object_update = object_update.reshape(B, V, T, 1, D)
C_object = self.mhc.post(
    object_update,
    residual=C_s,
    state=mhc_state,
)
C_object *= camera_state_valid[..., None]
```

`forward_update()` を使う点をtestで固定する。完全なblock outputをpostへ渡さない。

### 7.2 Spatial path

```text
C_spatial = permute(C_object, B,V,T,Q,D -> B,T,V,Q,D)
U = concat(S_s, reshape(C_spatial, B,T,VQ,D))
U = reshape(U, B*T,Q+VQ,D)
U' = spatial_block(U, spatial mask, spatial RoPE)
```

split後:

```text
S_spatial: [B,T,Q,D]
C_{s+1}:   [B,V,T,Q,D]
```

両方へ対応 validity maskを再適用する。

### 7.3 Query temporal path

```text
query_values = permute(S_spatial, B,T,Q,D -> B,Q,T,D)
query_values = reshape(B*Q,T,D)
```

Global stage:

```python
query_values = query_temporal_block(
    query_values,
    freqs_cis=time_freqs,
    attn_mask=query_temporal_attention_mask,
)
```

CSWA stage:

```python
query_values = query_temporal_block(
    query_values,
    freqs_cis=time_freqs,
    state_valid=query_temporal_state_valid,
)
```

これを `[B,T,Q,D]` へ戻し、frame maskを再適用して `S_{s+1}` とする。

## 8. Spatial RoPE

既存の time/camera/role 3-axis coordinate contractを維持する。ただし candidate axis `P` は常に `Q` となる。

```text
slot tokens:    (time=t, camera=0, role=0)
camera tokens:  (time=t, camera=v+1, role=1)
```

candidate index自体をRoPE coordinateへ追加しない。candidate streamsはset-likeであり、mHC coefficient generatorとspatial attentionのpermutation propertyを保つ。

## 9. Configuration

strict task configへ次を追加する。

```python
@dataclass(frozen=True, slots=True)
class TrackQueryMHCConfig:
    coefficient_dim: int
    sinkhorn_iters: int
    eps: float
    residual_identity_bias: float
    update_scale_init: float


@dataclass(frozen=True, slots=True)
class TrackQueryCSWAConfig:
    compression_ratio: int
    window_radius: int
    backend: Literal["reference", "cuda"]


@dataclass(frozen=True, slots=True)
class TrackQueryModelConfig:
    ...
    mhc: TrackQueryMHCConfig
    cswa: TrackQueryCSWAConfig
```

Hydra base example:

```yaml
mhc:
  coefficient_dim: 64
  sinkhorn_iters: 20
  eps: 1.0e-6
  residual_identity_bias: 4.0
  update_scale_init: 0.0

cswa:
  compression_ratio: 4
  window_radius: 4
  backend: reference
```

hidden dimension、head count、head dimension、RoPE dimensionは既存top-level configからcomponent configへ渡し、YAMLで重複管理しない。stage patternは設定項目にせず、`num_stages % 4` から固定する。

validation:

- `num_stages > 0 and num_stages % 4 == 0`
- `num_queries > 0`
- component固有値は各designのvalidationを満たす
- unknown/missing keyをstrict parserでreject
- CUDA backend指定時に利用環境をsilent fallbackしない

## 10. Model class structure

`BLCSTrackQueryModel` は巨大なforward loopへ全処理を埋め込まず、次の責務を持つ。

```text
- observation encoder construction
- initial slot embedding
- spatial/time frequency construction
- ModuleList[BLCSTrackQueryStage]
- final norm and output heads
```

各stageのshape変換、mHC、temporal modeは `BLCSTrackQueryStage` が所有する。modelはstage間stateを渡すだけにする。

## 11. Output and checkpoint contract

outputは変更しない。

```text
position:        [B,T,Q,3]
presence_logits: [B,T,Q]
```

既存 loss/decoder contractを維持する。architecture parameter/state dictは大きく変わるため、旧 track-query checkpointのstrict loadは失敗させる。key renameやmissing moduleをsilent補完するcheckpoint migrationを同時実装しない。

## 12. Files owned by this Implementer

主 ownership:

```text
src/tasks/blcs/data/tracking_dataset.py
src/tasks/blcs/data/tracking_augmentation.py
src/tasks/blcs/data/tracking_types.py
src/tasks/blcs/model_io/attention_masks.py
src/tasks/blcs/model_io/adapters.py
src/tasks/blcs/configuration.py
src/tasks/blcs/configs/model/_track_query.yaml
src/tasks/blcs/models/blcs_track_query_model.py
src/tasks/blcs/models/components/observation_fusion.py
src/tasks/blcs/models/components/track_query_stage.py
src/tasks/blcs/README.md
```

関連 unit/integration testsを所有する。Components 1–4のproduction filesおよびCUDA opsは変更しない。共有primitiveに不足があればparentへ明示的なinterface blockerとして返し、ownershipを越えて即修正しない。

## 13. Required tests

### Data/I/O

1. sample candidate axisが常にexact `Q`。
2. clip全体のphysical tracksが`Q`超でも、concurrent countが`Q`以下ならpack可能。
3. concurrent count `>Q` はrejectし、truncationしない。
4. target/observation assignmentの独立性と全view同期。
5. candidate mask、visibility、frame/view mask invariant。
6. collateがcandidate axisを可変paddingしない。
7. inference inputが`<Q`なら明示padding、`>Q`ならreject。
8. point-fusion maskがpadding candidateを除外。

### Configuration

9. nested config parseとunknown/missing key reject。
10. `num_stages`が4の倍数でない場合のreject。
11. backend/value range validation。

### Stage/model

12. stage type patternが `C,C,C,G` を全cycleで満たす。
13. object/query temporal modeが各stageで一致。
14. object pathが`forward_update()`を使い、full residual outputをpostへ渡さない。
15. shape trace: `[B,V,T,Q,D] -> [BV,T,D] -> [B,V,T,Q,D]`。
16. spatial sequence lengthが`Q+VQ`。
17. padding candidate値を変更してもvalid outputs不変。
18. padded frame/view outputsがzeroまたはmask外。
19. all-invisible frameを両 `mask_invisible_observations` modeで検証。
20. CPU reference forward/backward finite。
21. output key/shape/dtype/deviceの既存adapter回帰。
22. serializationとstrict old-checkpoint failure。
23. focused model smoke、ruff、mypy、関連pytest。

## 14. Completion handoff

terminal handoffには、変更path、data migration、shape trace、stage pattern、focused command結果、旧checkpoint非互換、残るCUDA/performance riskを記載する。CUDA高速化や学習品質改善を完了扱いにしない。
