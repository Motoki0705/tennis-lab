# BLCS / PLCS / SLCS 入力・マスク契約

この文書は `feat/issue-753-padding-mask-refactor` における、BLCS・PLCS・SLCS のモデル入力とマスクの横断契約をまとめる。タスク固有のモデル構成や実行方法は、[BLCS README](../src/tasks/blcs/README.md)、[PLCS README](../src/tasks/plcs/README.md)、[SLCS README](../src/tasks/slcs/README.md) を正本とする。この文書が所有するのは、タスク間で混同しやすい名前、shape、極性、派生関係、Datasetからモデル入力までの流れ、旧契約からの移行である。

## 記号

| 記号 | 意味 |
|---|---|
| `B` | batch size |
| `V` | camera / view 数 |
| `T` | temporal window の frame 数 |
| `Q` | fixed-query / lifecycle slot 数 |
| `P_phys` | Datasetがsceneから読むpacking前のphysical track数 |
| `P` | SLCS の player slot 数 |
| `J` | player joint 数。現行は COCO17 の `17` |
| `K` | court keypoint 数。profile / config で決まり、tracking BLCS/PLCS は現行 `14` |
| `T_d` | SLCS window 内の DINO sample 数 |
| `S` | 1 DINO sample の patch token 数 |
| `C` | DINO embedding 幅 |
| `E` | SLCS entity 数。`P + 1`（players + ball） |

2D 座標は normalized image UV で、末尾軸は `(u, v)` である。BLCS/PLCS の adapter は `[0, 1]` を要求する。SLCS も normalized UV を契約とし、visible な値だけを adapter の許容範囲内か検証する。3D target / prediction は `COURT_COORD_SCALE_XYZ` で正規化したコート座標、player rotation は yaw の `(cos, sin)` である。物理単位への変換は [`src/utils/schema/court.py`](../src/utils/schema/court.py) を正本とする。

## 最初に確認する極性

| タスク | 公開mask | `True` / 正値の意味 | `False` / 0 の意味 |
|---|---|---|---|
| BLCS / PLCS / SLCS | `padding_mask` | padding。attention・lossから除外する候補 | 実在する view/frame context |
| 全タスク | `*_vis` / `*_kp_vis` | 観測できた点・object | 不可視。paddingとは限らない |
| SLCS | `player_valid` | そのplayerの2D pose観測がある | pose観測がない |
| SLCS | `dino_padding_mask` | DINO sample用のpadding slot | 対応するsparse DINO sampleがある |
| 内部attention | `*_attention_keep_mask` / `*_attn_mask` | attention可能な query-key pair | attention禁止 |
| 学習target | `target_*_valid`, `target_presence`, `target_slot_mask` | label / instance / slotが学習対象 | 対象外 |

最重要ルールは次の3点である。

1. BLCS/PLCS/SLCS の `padding_mask` はすべて `True=padding` である。SLCSのsparse DINO軸だけは別tensorの `dino_padding_mask` を使う。
2. visibility は観測表現を選ぶための情報であり、padding ではない。非padding contextの不可視tokenは learned invisible token としてattentionに参加する。
3. 外部callerは内部attention maskを組み立てない。公開padding / validityから adapter またはmodelが一意に導出する。

```text
view/time padding ──> context/frame validity ──> attention keep-mask ──> state処理・loss除外
observation visibility ────────────────────────> visible / invisible tokenの選択
target validity / presence ───────────────────> loss・matching・metricsの対象選択
```

UVが0であること自体はpaddingや不可視を意味しない。必ず対応するvisibility / validityを参照する。

## BLCS

### 公開モデル入力

BLCS の全profileは `ball_uv`, `ball_vis`, `court_kp`, `court_vis`, `padding_mask` の5入力に統一されている。境界検証の正本は [`src/tasks/blcs/model_io/adapters.py`](../src/tasks/blcs/model_io/adapters.py) である。

| profile | `ball_uv` | `ball_vis` | `court_kp` | `court_vis` | `padding_mask` |
|---|---|---|---|---|---|
| single | `(B,T,2)` | `(B,T)` | `(B,K,2)` | `(B,K)` | `(B,T)` |
| multiview / axial | `(B,V,T,2)` | `(B,V,T)` | `(B,V,T,K,2)` | `(B,V,T,K)` | `(B,V,T)` |
| track-query | `(B,V,T,Q,2)` | `(B,V,T,Q)` | `(B,V,T,14,2)` | `(B,V,T,14)` | `(B,V,T)` |

multiview / axial adapter は static court 入力 `(B,V,K,2)` / `(B,V,K)` も受け、model callでは `T` 軸へ展開する。UVはfloating tensor、`padding_mask`は必ず `torch.bool` である。standard profileのvisibilityはbinaryな数値tensorも受けてboolへ正規化するが、track-queryではvisibilityも `torch.bool` を要求する。

track-queryの観測幅は常に `Q=model.num_queries` である。collateがpaddingするのは `V` / `T` 軸だけで、`Q` 軸のpadding maskは存在しない。推論で観測候補数がQ未満の入力を受ける専用predictorは、残りをzero UVかつ `ball_vis=False` のinvisible queryとして埋める。観測候補数がQを超える入力は拒否する。

`ball_vis=False` は「そのqueryのボールが観測できない」を表し、contextがpaddingであることを表さない。非padding contextならattentionに残る。`court_vis`も同様に、不可視court pointの座標を無効化するために使い、view/frame paddingには使わない。

### paddingからの派生

single / multiview / axial は [`src/tasks/blcs/models/components/padding.py`](../src/tasks/blcs/models/components/padding.py) でprofile別maskを導出する。

| profile | 主な派生値 | shape / 意味 |
|---|---|---|
| single | `frame_valid` | `(B,T) = ~padding_mask` |
| single | `attention_keep_mask` | `(B,K+T,K+T)`、court + ball self-attention |
| multiview | `context_valid` | `(B,V,T) = ~padding_mask` |
| multiview | `frame_valid` | `(B,T) = context_valid.any(V)` |
| multiview | `cross_attention_keep_mask` | `(B*T,1,V*(K+1))`、frame内のcamera tokenへのcross-attention |
| multiview | `query_attention_keep_mask` | `(B,T,T)`、trajectory queryのtime attention |
| axial | `camera_attention_keep_mask` | `(B*T,V,V)` |
| axial | `time_attention_keep_mask` | `(B*V,T,T)` |
| axial | `sliding_attention_keep_mask` | `(B*V,T,T)`、time-local windowも反映 |

track-queryはPLCSと共有する [`build_fixed_query_padding_masks()`](../src/utils/models/multiview_padding.py) をmodel内で呼ぶ。詳細は「共有fixed-query契約」を参照する。

standard trajectoryのloss maskはsingleで `~padding_mask`、multiview / axialで `(~padding_mask).any(dim=1)` である。track-queryも同じframe validityをmatching、presence、position、smoothness、gravity、metricsへ渡す。reprojection lossはさらにtarget visibilityとcamera前方条件を適用する。

### model callに渡さない値

`position_3d`, `velocity_3d`, camera parameters、clean augmentation targetは学習・reprojection用であり、5入力のmodel callには含めない。track-queryの `target_position`, `target_velocity`, `target_presence`, `target_instance_id`, `target_slot_mask`, `clean_ball_*`, `candidate_gt_index` も学習・debug用である。

## PLCS

### standard profileの公開入力

Dataset / DataModuleのcanonical batchは次の5入力を持つ。境界検証の正本は [`src/tasks/plcs/model_io/adapters.py`](../src/tasks/plcs/model_io/adapters.py) である。

| 入力 | canonical shape | 内容 |
|---|---|---|
| `human_kp` | `(B,V,T,17,2)` | player poseのnormalized UV |
| `human_vis` | `(B,V,T,17)` | joint visibility |
| `court_kp` | `(B,V,T,K,2)` | court keypointのnormalized UV |
| `court_vis` | `(B,V,T,K)` | court point visibility |
| `padding_mask` | `(B,V,T)` bool | `True=padding` |

adapterはcomposition時に決まったprofileへだけ変換する。

| profile | model-ready layout | paddingの利用 |
|---|---|---|
| frame | 選択camera・1 frameを `(B,...)` にする | forwardでは未使用。loss / metricsで除外 |
| sequence | 選択cameraを `(B*T,...)` にflattenし、decode時に `(B,T,...)` へ戻す | frame model forwardでは未使用。loss / metricsで除外 |
| multiview | canonical `(B,V,T,...)` のまま | model / axial attentionとloss / metricsで利用 |

standard PLCSでは `K` はconfigで決まる。通常の合成データはCourtKP20を使う。visibilityはbinaryなbool / 数値tensorを受けるが、`padding_mask`は `torch.bool` 固定である。

### track-queryの公開入力

| 入力 | shape | dtype / 意味 |
|---|---|---|
| `human_kp` | `(B,V,T,Q,17,2)` | floating normalized UV |
| `human_vis` | `(B,V,T,Q,17)` | bool、joint visibility |
| `court_kp` | `(B,V,T,14,2)` | floating normalized UV |
| `court_vis` | `(B,V,T,14)` | bool、court visibility |
| `padding_mask` | `(B,V,T)` | bool、`True=padding` |

`Q=model.num_queries` は固定で、BLCSと同様にQ軸をpaddingしない。全jointが不可視の非padding slotはlearned invisible tokenになり、camera / time文脈から更新される。`detection_gt_index` はaugmentation / debug用でmodelへ渡さない。

track-queryのtargetは `target_position (B,T,Q,3)`, `target_rotation (B,T,Q,2)`, `target_presence (B,T,Q)`, `target_instance_id (B,T,Q)`, `target_slot_mask (B,Q)` である。inactive instance IDは `-1` とする。これらはmodel入力ではない。

### paddingからの派生

- `PLCSMultiViewModel` は `~padding_mask` からcamera/frame token validityとpairwise attention keep-maskをmodel内で構築する。
- axial familyはadapterの [`prepare_axial_attention_masks()`](../src/tasks/plcs/model_io/attention_masks.py) が `(B*T,V,V)` のcamera maskと `(B*V,T,T)` のtime maskを作る。
- track-queryは共有fixed-query utilityと共有`FixedQueryTrackStage`をmodel内で使い、mHC object temporal、global spatial、query temporalを順に適用してstate / outputをpadding位置で再度0化する。temporal modeは `CSWA, CSWA, CSWA, Global MHA` の固定cycleである。
- standard loss / metricsはframe / sequence profileでは選択cameraのpaddingを、multiview profileでは全viewがpaddingのframeを除外する。track-queryは1 view以上が実contextであるframe validityに `target_presence` / `target_slot_mask` を重ねてmatchingとlossを計算する。

## Track-query Datasetから5入力まで

この節はBLCS / PLCSの学習・validation・test用track-query経路を扱う。SLCSにはtrack-query Datasetはない。両タスクとも、Datasetが返す全fieldをmodelへ渡すのではなく、collate後にtask-local adapterが5入力だけを `ModelCall` へ射影する。

```text
split file
  -> TrackingDataModule
  -> TrackingDataset.__getitem__
       -> scene読込
       -> clip / view選択
       -> physical trackをfixed Qへlifecycle packing
       -> clean sample作成
       -> trainだけ観測augmentation
  -> tracking collate（V/Tだけpadding）
  -> tracking LightningModule
  -> task-local adapter（入力・target検証）
  -> ModelCall.kwargs（5 tensorだけ）
  -> BoundModelIO.execute_call()
  -> track-query model.forward()
```

### 1. DataModuleからsample生成まで

通常backendでは [`SceneDirectoryDataModule`](../src/tasks/base/data/datamodule.py) が `train.txt`, `val.txt`, `test.txt` からtask固有Datasetを作る。trainだけ `augment=True`、val/testは `False` である。BLCSは [`BLCSTrackingDataModule`](../src/tasks/blcs/data/tracking_datamodule.py)、PLCSは [`PLCSTrackingDataModule`](../src/tasks/plcs/data/tracking_datamodule.py) がtracking専用Datasetとcollateを選ぶ。

共通の [`SceneDatasetBase.__getitem__()`](../src/tasks/base/data/scene_dataset.py) の順序は固定されている。

```text
_load_scene() -> build_sample() -> augment_sample()
```

[`CanonicalTrackingDataset`](../src/tasks/base/data/canonical_tracking.py) はconfigからclip長、view数、`Q=model.num_queries`、slot再利用gap、train時のslot randomizationを受け持つ。`build_sample()` 内ではtrainがrandom clip/view、val/testがdeterministicなcenter clip/viewを選ぶ。

### 2. physical trackからfixed-Q sampleへ

両Datasetはsceneのphysical track軸 `P_phys` を、そのままmodelへ渡さない。`physical_presence (T,P_phys)` の連続birth/death intervalを [`build_fixed_lifecycle_assignment()`](../src/tasks/base/data/lifecycle_slots.py) で再利用可能なQ slotへ割り当てる。同時に必要なslot数がQを超える場合は切り捨てず例外にする。

| 段階 | BLCS | PLCS |
|---|---|---|
| Dataset | [`BLCSTrackingDataset`](../src/tasks/blcs/data/tracking_dataset.py) | [`PLCSTrackingDataset`](../src/tasks/plcs/data/tracking_dataset.py) |
| physical target | `ball_pos_norm`, `ball_vel_world`, `ball_present` | `position`, `rotation`, pose targets, `person_present` |
| physical observation | cameraごとの `ball_uv (T,P_phys,2)`, `ball_vis (T,P_phys)` | cameraごとの `human_kp_uv (T,P_phys,17,2)`, `human_kp_vis (T,P_phys,17)` |
| lifecycle assignment | targetとobservationで別々に構築 | 1つのassignmentをtargetとobservationで共有 |
| packed observation | `ball_uv (V,T,Q,2)`, `ball_vis (V,T,Q)` | `human_kp (V,T,Q,17,2)`, `human_vis (V,T,Q,17)` |
| packed target | position / velocity / presence / instance ID | position / rotation / pose / presence / instance ID |

BLCSのobservation packingは [`pack_observation_candidates()`](../src/tasks/blcs/data/observation_candidates.py) が全viewで同期したassignmentを作る。trainではtarget assignmentとobservation assignmentが独立にslot permutationを引くため、同じphysical instanceが両者で同じQ列に入るとは限らない。PLCSは1つのassignmentの `pack_tensor()` を観測とtargetの両方へ使う。

各sampleの `padding_mask (V,T)` はこの時点では全 `False` である。Datasetが選んだclip/viewはすべて実contextであり、batch間のV/T差を埋めるpaddingはcollateで初めて追加する。court入力は両タスクともtracking model向けの先頭14点を使う。

Datasetが作るfieldは3種類に分かれる。

| 分類 | BLCS | PLCS |
|---|---|---|
| model候補 | `ball_uv`, `ball_vis`, `court_kp`, `court_vis`, `padding_mask` | `human_kp`, `human_vis`, `court_kp`, `court_vis`, `padding_mask` |
| 学習target | `target_position`, `target_velocity`, `target_presence`, `target_instance_id`, `target_slot_mask` | `target_position`, `target_rotation`, pose targets, `target_presence`, `target_instance_id`, `target_slot_mask` |
| augmentation / debug | `clean_ball_uv`, `clean_ball_vis`, `candidate_gt_index` | `clean_human_kp`, `clean_human_vis`, `detection_gt_index` |

`clean_*` はdisk上の未加工physical配列ではなく、fixed-Qへpackingした後、augmentationする前のcloneである。

### 3. train時の観測augmentation

augmentationはmodel observationだけを変更し、target、clean field、`padding_mask`、Q順を保持する。

- BLCSの [`BLCSTrackingCandidateAugmentation`](../src/tasks/blcs/data/tracking_augmentation.py) は `(V,T,Q,2)` を一時的に `(V*Q,T,2)` へ変形し、single-ball augmentationを各candidateへ適用して元shapeへ戻す。
- PLCSの [`PLCSTrackingDetectionAugmentation`](../src/tasks/plcs/data/tracking_augmentation.py) は `(V,T,Q,17,2)` のQ/joint軸を一時的にまとめ、pose observation augmentation後に元shapeへ戻す。
- 両者ともcourt inputをcloneして保持する。visibility dropoutやfalse positiveは `ball_vis` / `human_vis` を変えても、それをview/time paddingへ昇格させない。

したがって、augmentation後も `padding_mask=False, visibility=False` は正当な組み合わせであり、「実contextだが観測できないtoken」を意味する。

### 4. collateでV/Tだけをpadding

[`collate_blcs_tracking_batch()`](../src/tasks/blcs/data/tracking_dataset.py) と [`collate_plcs_tracking_batch()`](../src/tasks/plcs/data/tracking_dataset.py) は、共通の [`pad_and_stack_tracking_batch()`](../src/tasks/base/data/canonical_tracking.py) でbatch内最大 `V_max`, `T_max` へ末尾paddingする。

| task | sample | collated batch |
|---|---|---|
| BLCS | `ball_uv (V,T,Q,2)` | `ball_uv (B,V_max,T_max,Q,2)` |
| BLCS | `ball_vis (V,T,Q)` | `ball_vis (B,V_max,T_max,Q)` |
| PLCS | `human_kp (V,T,Q,17,2)` | `human_kp (B,V_max,T_max,Q,17,2)` |
| PLCS | `human_vis (V,T,Q,17)` | `human_vis (B,V_max,T_max,Q,17)` |
| 共通 | `court_kp (V,T,14,2)` | `court_kp (B,V_max,T_max,14,2)` |
| 共通 | `court_vis (V,T,14)` | `court_vis (B,V_max,T_max,14)` |
| 共通 | `padding_mask (V,T)` | `padding_mask (B,V_max,T_max)` |

追加領域はUV / targetが0、visibilityが `False`、`padding_mask` が `True`、instance/debug IDが `-1` になる。targetのT軸も `T_max` へ揃えるが、Q軸はDataset時点でexact Qなのでpaddingしない。

### 5. batchから5入力への射影

tracking LightningModuleはmodelを直接 `model(**batch)` で呼ばない。BLCSは [`TrackQueryModelIOAdapter.build_training_batch()`](../src/tasks/blcs/model_io/adapters.py)、PLCSは [`PLCSTrackQueryIOAdapter.prepare_training_batch()`](../src/tasks/plcs/model_io/adapters.py) を必ず通す。

adapterは次を別々に行う。

1. 5つのmodel入力についてkey、dtype、rank、shape、normalized UV、Q幅を検証する。
2. training targetのshape、finite値、inactive instance IDの `-1` sentinelを検証する。
3. target / clean / debug fieldを除外し、5入力だけをimmutable `ModelCall.kwargs`へ格納する。

| task | 最終 `ModelCall.kwargs` |
|---|---|
| BLCS | `ball_uv`, `ball_vis`, `court_kp`, `court_vis`, `padding_mask` |
| PLCS | `human_kp`, `human_vis`, `court_kp`, `court_vis`, `padding_mask` |

[`BoundModelIO.execute_call()`](../src/tasks/base/model_io/contracts.py) だけが検証済みcallを `model(*args, **kwargs)` として実行する。modelは受け取った `padding_mask` から内部state validity / attention keep-maskを作るため、Dataset、collate、LightningModuleはattention maskを生成しない。

### 6. backend・推論入口の差

- default trackingは固定 `scene_dir` のtrain/val/testを上記経路で読む。
- chunked trackingも同じDataset、augmentation、collate、adapterを使う。差はtrainのscene sourceだけで、val/testは固定 `scene_dir` のままである。
- tracking predictorはDataset/DataModuleを通らず、callerがbatch化済み5入力を直接渡す。PLCSはexact Qを要求する。BLCSの高水準predictorだけは観測候補数がQ未満ならzero UV / `ball_vis=False` でQへ埋めてからstrict adapterへ渡す。このQ補完は `padding_mask` ではない。

## 共有fixed-query契約

BLCS / PLCS track-queryは [`src/utils/models/multiview_padding.py`](../src/utils/models/multiview_padding.py) を共有する。入力は `padding_mask (B,V,T)`、`True=padding` と固定幅 `Q` だけであり、visibilityは受け取らない。

| 派生値 | shape | `True` の意味 |
|---|---|---|
| `context_valid` | `(B,V,T)` | そのcamera/frameが実context |
| `frame_valid` | `(B,T)` | 1 view以上が実context |
| `object_state_valid` | `(B,V,T,Q)` | camera/frame上の各fixed object stateが有効 |
| `spatial_attention_keep_mask` | `(B*T,Q+V*Q,Q+V*Q)` | query / camera token間のattentionを許可 |
| `object_temporal_state_valid` | `(B*V,T)` | cameraごとのobject-temporal stateが有効 |
| `object_temporal_attention_keep_mask` | `(B*V,T,T)` | object-temporal attentionを許可 |
| `query_temporal_state_valid` | `(B*Q,T)` | queryごとのtemporal stateが有効 |
| `query_temporal_attention_keep_mask` | `(B*Q,T,T)` | query-temporal attentionを許可 |

BLCS/PLCSは同じ派生値をすべて使用し、共有`FixedQueryTrackStage`で `mHC object temporal -> global spatial -> query temporal` を実行する。ball/playerのvisibilityはこの契約へ入れず、task固有embeddingでvisible/invisible tokenを選択する。

[`build_self_attn_mask()`](../src/utils/models/transformer_utils.py) は全要素invalidの系列でsoftmaxのNaNを避けるため、内部的にtoken 0だけをvalidに修復する。この修復は入力を実データへ変換しない。raw validityはinvalidのままであり、各modelはstate / outputを再度0化する。

## SLCS

SLCSは単眼windowとsparse DINO tokensを使い、Issue 753で追加されたBLCS/PLCSの `padding_mask` 共通utilityは使用しない。公開入力の正本は [`src/tasks/slcs/model_io/adapter.py`](../src/tasks/slcs/model_io/adapter.py) と [`src/tasks/slcs/data/types.py`](../src/tasks/slcs/data/types.py) である。

| 入力 | shape | dtype / 意味 |
|---|---|---|
| `player_kp` | `(B,P,T,17,2)` | float32 normalized UV |
| `player_kp_vis` | `(B,P,T,17)` | float32 `[0,1]` visibility score |
| `player_valid` | `(B,P,T)` | bool。`(player_kp_vis > 0).any(-1)` と完全一致 |
| `ball_uv` | `(B,T,2)` | float32 normalized UV |
| `ball_vis` | `(B,T)` | bool、ball observation visibility |
| `court_kp` | `(B,T,K,2)` | float32 normalized UV |
| `court_vis` | `(B,T,K)` | float32 `[0,1]` visibility score |
| `padding_mask` | `(B,T)` | bool、`True=window右側のpadding` |
| `dino_tokens` | `(B,T_d,S,C)` | float32 sparse patch tokens |
| `dino_frame_idx` | `(B,T_d)` | int64 window-relative frame index |
| `dino_padding_mask` | `(B,T_d)` | bool、`True=DINO sample用padding slot` |

`padding_mask` は最低1つの実frameを残し、`False...False, True...True` の連続padding suffixでなければならない。`dino_padding_mask`も同じsuffix規則を持つ。`player_valid`と`ball_vis`はpadding frameをvalidにできず、非paddingの `dino_frame_idx` は `[0,T)` 内で厳密な昇順とする。

visibilityは座標の0化とvisible / invisible tokenの選択に使う。attention構造は観測visibilityではなく `padding_mask` から作るため、実frame上の不可視player / ball tokenもattentionへ参加する。

### SLCS内部mask

| 派生値 | shape | 導出 |
|---|---|---|
| `entity_attention_keep_mask` | `(B*T,E,E)` | `~padding_mask` を `E=P+1` entityへ展開 |
| `time_attention_keep_mask` | `(B*E,T,T)` | 同じframe validityをentityごとのtime軸へ変形 |
| `dino_attention_keep_mask` | `(B,T*E,T_d*S')` | `~dino_padding_mask` をDINO encoder後のpatch数 `S'` へ展開 |
| `dino_batch_has_evidence` | `(B,)` | 有効なDINO keyが1つ以上ある |

DINO evidenceがないsampleでは数値安定性のためkey 0だけを内部keepにするが、`dino_batch_has_evidence=False` によりDINO更新を無効化する。

SLCSの教師maskは観測maskと別物である。`target_player_valid` はpseudo-label品質、`target_ball_valid` はball pseudo-label品質を表す。adapterは次を作る。

```text
player_mask = target_player_valid & ~padding_mask[:, None, :]
ball_mask   = target_ball_valid   & ~padding_mask
```

supervised lossとmetricsはこのmaskにconfidence weightを重ねる。smoothness / ground penetrationはlabel validityではなく `~padding_mask` の実frameを使う。

## visibility・padding・教師maskの使い分け

| 質問 | 参照する値 |
|---|---|
| このview/frameはcollateで追加されたか | 全taskの `padding_mask` |
| 2D点が観測できたか | `ball_vis`, `human_vis`, `player_kp_vis`, `court_vis` |
| SLCS player pose観測が1 joint以上あるか | `player_valid` |
| track-queryの物理instanceがactiveか | `target_presence` |
| lifecycle target slotをmatching対象にするか | `target_slot_mask` |
| pseudo-labelをsupervised lossへ入れるか | SLCS: `target_player_valid`, `target_ball_valid` |
| DINO sampleがbatch paddingか | `dino_padding_mask` |
| attention pairを許可するか | 内部の `*_attention_keep_mask` / `*_attn_mask` |

## Issue 753以前の入力からの移行

旧モデル入力keyのうち下表の該当keyはadapter境界で明示的にrejectする。旧設定keyはconfiguration境界でunknown keyとしてrejectし、旧disk名は新しい名前へfallbackしない。caller側で必要な変換を行う。

| 旧契約 | 新契約 |
|---|---|
| BLCS `ball_mask` (`True=valid`) | `padding_mask = ~ball_mask.bool()` |
| PLCS `human_mask` (`True=valid`) | `padding_mask = ~human_mask.bool()` |
| BLCS/PLCS tracking `frame_mask` + `view_mask` (`True=valid`) | `padding_mask = ~(view_mask[:, :, None] & frame_mask[:, None, :])` |
| BLCS `ball_visible` | `ball_vis` |
| BLCS `candidate_mask` | 廃止。Q幅を固定し、非観測queryは `ball_vis=False`。Q軸をpadding扱いしない |
| PLCS `detection_mask` | 廃止。joint単位の `human_vis` を使い、全joint不可視でも非paddingならattentionへ残す |
| `mask_invisible_observations` | 廃止。不可視tokenをattentionへ残す挙動に固定 |
| caller生成のBLCS attention mask | 廃止。5入力からmodel内で導出 |
| SLCS `frame_mask` (`True=valid`) | `padding_mask = ~frame_mask`。旧keyはreject |
| SLCS `dino_valid` (`True=valid`) | `dino_padding_mask = ~dino_valid`。旧keyはreject |
| BLCS/PLCS camera arrayの `*_visible.npy` | `cam_{i}_ball_vis.npy`, `cam_{i}_human_kp_vis.npy`, `cam_{i}_court_kp_vis.npy` などの `*_vis.npy`。旧名へのfallbackはない |

SLCSの旧評価artifactもaliasでは読まない。`frame_mask`を保存した`.npz`は新しい`padding_mask=True`契約で再生成する。

旧checkpointは公開入力名だけでなくarchitectureも異なる場合がある。自動key migrationやmissing parameter補完を前提にせず、strict load errorとして扱う。

## 実装・テスト対応表

| 契約 | 実装 | 主なテスト |
|---|---|---|
| BLCS公開5入力 | [`blcs/model_io/adapters.py`](../src/tasks/blcs/model_io/adapters.py) | [`test_adapters.py`](../tests/unit/tasks/blcs/model_io/test_adapters.py), [`test_padding_contract.py`](../tests/unit/tasks/blcs/models/test_padding_contract.py) |
| BLCS standard padding派生 | [`blcs/models/components/padding.py`](../src/tasks/blcs/models/components/padding.py) | [`test_padding.py`](../tests/unit/tasks/blcs/models/components/test_padding.py) |
| PLCS公開入力 | [`plcs/model_io/adapters.py`](../src/tasks/plcs/model_io/adapters.py) | [`test_adapters.py`](../tests/unit/tasks/plcs/model_io/test_adapters.py) |
| fixed-query共通padding | [`utils/models/multiview_padding.py`](../src/utils/models/multiview_padding.py) | [`test_multiview_padding.py`](../tests/unit/utils/models/test_multiview_padding.py) |
| BLCS visibility / padding分離 | [`blcs_track_query_model.py`](../src/tasks/blcs/models/blcs_track_query_model.py) | [`test_blcs_track_query_model.py`](../tests/unit/tasks/blcs/models/test_blcs_track_query_model.py) |
| PLCS visibility / padding分離 | [`plcs_track_query_model.py`](../src/tasks/plcs/models/plcs_track_query_model.py) | [`test_plcs_track_query_model.py`](../tests/unit/tasks/plcs/models/test_plcs_track_query_model.py) |
| tracking Datasetから5入力 | [`blcs/data/tracking_dataset.py`](../src/tasks/blcs/data/tracking_dataset.py), [`plcs/data/tracking_dataset.py`](../src/tasks/plcs/data/tracking_dataset.py) | [`test_training_smoke.py`](../tests/integration/tasks/tracking/test_training_smoke.py) |
| SLCS入力・派生mask | [`slcs/model_io/adapter.py`](../src/tasks/slcs/model_io/adapter.py), [`slcs/models/components/padding.py`](../src/tasks/slcs/models/components/padding.py) | [`test_adapter.py`](../tests/unit/tasks/slcs/model_io/test_adapter.py), [`test_padding.py`](../tests/unit/tasks/slcs/models/components/test_padding.py) |
