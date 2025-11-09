# イベント検知モデル アーキテクチャ設計（最終仕様）

本書は、シーンモデルの設計方針に整合する形で**イベント検知モデル**の最終仕様を確定する。学習方針・損失設計・後処理（平滑化・ピーク検出）は別文書とする。

---

## 0. 要旨（確定事項）

* **入力**: シーンモデルの `obj_query [B,T,Q_obj,D_s]`、時系列文脈 `z [B,T,D_s]`、（任意）音声潜在 `audio_tokens [B,T_a,D_a]`。
* **K/V**: **`obj_query` のみ**を使用（付加特徴の連結や加算は行わない）。LayerNorm とマスク以外で改変しない。
* **クエリ**: `Q_ev` 本のイベント探索クエリ。`E` クラスとは独立。
* **初期化**: `E_prev` と `E_init` を**softmax 凸結合**し、`z_t` と **絶対時刻** `AbsTimePE(t)` を加算（任意で音声バイアス）。
* **参照**: ローカル時間窓 `[t−k, t+k]` 上の **(slot, time)** 離散格子に対する**疎参照**。時間は**離散フレーム**のみ（サブフレーム補間は実施しない）。
* **相対時刻**: `RelTimePE(Δ)` を**重み生成**に注入し、ロジットに距離ペナルティ `−λ|Δ|` を適用。
* **オフセット生成単位**: 既定は **τ（参照フレーム）ごと**（`offset_mode="per_tau"`）。
* **プーリング**: **廃止**。デコーダ出力の**各イベントクエリ**を**そのままヘッド**へ入力し、`[B,T,Q_ev,E]` を出力。
* **スコープ外**: 持続イベント（開始/終了）、サブフレーム精度（δt 回帰）、事前スコア（xyz 等）の利用。
* **動作**: オフライン前提（未来参照可）。

---

## 1. 記号・I/O

* バッチ `B`、フレーム `T`、シーン側スロット `Q_obj`、イベントクエリ `Q_ev`、次元 `D_s`（入力）/`D_h`（内部）。
* 窓半径 `k`、窓長 `F=2k+1`、サンプル点 `M`（slot内サンプル数）、`topK_tau`（近傍フレーム数）、`topK_slot`（近傍スロット数）。

### 入力

* `obj_query`: `[B, T, Q_obj, D_s]`
* `temporal_tokens`（= `z`）: `[B, T, D_s]`
* `audio_tokens`（任意）: `[B, T_a, D_a]`

### 出力

* `logits`: `[B, T, Q_ev, E]`
* `prob`: `[B, T, Q_ev, E]` （sigmoid）

---

## 2. コンフィグ API

```python
from dataclasses import dataclass

@dataclass
class EventConfig:
    D_s: int          # シーン側次元
    D_h: int          # イベント内部次元（通常 D_h=D_s）
    num_events: int   # E

    # クエリ/デコーダ
    num_event_queries: int  # Q_ev
    ev_decoder_layers: int
    window_k: int           # k
    num_points: int         # M（slotサンプル数）
    topk_tau: int           # 近傍フレーム数（<=2k+1）
    topk_slot: int          # 近傍スロット数（<=Q_obj）
    offset_mode: str = "per_tau"  # 既定
    tbptt_detach: bool = True

    # 時刻PE
    fps: float
    absK: int; abs_Thorizon: float
    relQ: int; rel_wmin: float; rel_wmax: float

    # 音声（任意）
    use_audio: bool = False
    D_a: int = 0
    beta_init: float = 0.0    # 学習可能遅延 β_global の初期値（秒）
```

---

## 3. モジュール構成とデータフロー

### 3.1 トップレベル

```python
class EventModel(nn.Module):
    def __init__(self, cfg: EventConfig):
        super().__init__()
        D_h = cfg.D_h
        # 時刻PE
        self.abs_pe   = AbsTimePE(K=cfg.absK, fps=cfg.fps, Thorizon=cfg.abs_Thorizon)
        self.abs_proj = nn.Linear(2*cfg.absK, D_h)
        self.rel_pe   = RelTimePE(Q=cfg.relQ, wmin=cfg.rel_wmin, wmax=cfg.rel_wmax)
        self.rel_proj = nn.Linear(2*cfg.relQ, D_h)

        # 入力射影
        self.z_proj   = nn.Linear(cfg.D_s, D_h)

        # 音声（任意）
        self.use_audio = cfg.use_audio
        if self.use_audio:
            self.a_align = EventTimeAlign(mode="interp")
            self.beta = nn.Parameter(torch.tensor(cfg.beta_init))  # β_global [sec]
            self.a_proj = nn.Linear(cfg.D_a, D_h)

        # デコーダ
        self.decoder = EventQueryDecoder(
            dim=D_h,
            num_event_queries=cfg.num_event_queries,
            num_layers=cfg.ev_decoder_layers,
            num_points=cfg.num_points,
            window_k=cfg.window_k,
            topk_tau=cfg.topk_tau,
            topk_slot=cfg.topk_slot,
            offset_mode=cfg.offset_mode,
            tbptt_detach=cfg.tbptt_detach,
            abs_pe=self.abs_pe, abs_proj=self.abs_proj,
            rel_pe=self.rel_pe, rel_proj=self.rel_proj,
            fps=cfg.fps,
        )

        # ヘッド（プーリング無し）
        self.head = EventHead(D_h, cfg.num_events)

    def forward(self, obj_query, temporal_tokens, audio_tokens=None):
        B,T,Q_obj,D_s = obj_query.shape
        z = self.z_proj(temporal_tokens)  # [B,T,D_h]

        if self.use_audio and audio_tokens is not None:
            a = self.a_align(audio_tokens, T)             # [B,T,D_a]
            a = time_shift_seconds(a, self.beta, fps=self.decoder.fps)  # β_global
            a = self.a_proj(a)                            # [B,T,D_h]
        else:
            a = torch.zeros(B,T,self.decoder.dim, device=z.device)

        ev_q = self.decoder(obj_query, z, a)              # [B,T,Q_ev,D_h]
        prob, logits = self.head(ev_q)                    # [B,T,Q_ev,E]
        return prob, logits
```

### 3.2 時刻埋め込み

```python
class AbsTimePE(nn.Module):
    def __init__(self, K, fps, Thorizon):
        super().__init__()
        self.phi = nn.Parameter(torch.rand(K) * 2*math.pi)
        self.scale = nn.Parameter(torch.tensor(1.0))
        wmax = 2*math.pi*0.45*fps
        wmin = 2*math.pi/Thorizon
        self.register_buffer('omega', torch.logspace(math.log10(wmin), math.log10(wmax), K), persistent=False)
    def forward(self, t_sec):
        # t_sec: float or tensor [...]
        tau = t_sec
        arg = self.scale * self.omega * tau + self.phi
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)  # [..., 2K]

class RelTimePE(nn.Module):
    def __init__(self, Q, wmin, wmax):
        super().__init__()
        self.register_buffer('omega', torch.logspace(math.log10(wmin), math.log10(wmax), Q), persistent=False)
    def forward(self, delta_sec):
        arg = self.omega * delta_sec[..., None]
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)  # [..., 2Q]
```

### 3.3 デコーダ（再帰 + slot×time 疎参照）

```python
class EventQueryDecoder(nn.Module):
    def __init__(self, dim, num_event_queries, num_layers, num_points, window_k,
                 topk_tau, topk_slot, offset_mode, tbptt_detach,
                 abs_pe, abs_proj, rel_pe, rel_proj, fps):
        super().__init__()
        self.dim = dim; self.Q_ev = num_event_queries; self.L = num_layers
        self.M = num_points; self.k = window_k
        self.topk_tau = topk_tau; self.topk_slot = topk_slot
        self.offset_mode = offset_mode; self.tbptt = tbptt_detach
        self.abs_pe = abs_pe; self.abs_proj = abs_proj
        self.rel_pe = rel_pe; self.rel_proj = rel_proj; self.fps = fps

        self.init_event_queries = nn.Parameter(torch.randn(self.Q_ev, dim))
        self.merge = EventQueryMergeLayer(dim)
        self.layers = nn.ModuleList([
            EventDeformableDecoderLayer(dim, num_points, offset_mode)
            for _ in range(self.L)
        ])

    def forward(self, obj_query, z, a):  # [B,T,Q_obj,D_s->D_h], [B,T,D_h], [B,T,D_h]
        B,T,Q_obj,D = obj_query.shape
        E_prev = self.init_event_queries.unsqueeze(0).expand(B, self.Q_ev, D)
        outs = []
        for t in range(T):
            if self.tbptt: E_prev = E_prev.detach()
            # クエリ初期化
            e_abs = self.abs_proj(self.abs_pe(torch.tensor(t/self.fps, device=z.device)))  # [2K]->[D]
            E_init = self.init_event_queries.unsqueeze(0).expand_as(E_prev)
            E_t0 = self.merge(E_prev, E_init, ctx=z[:, t], time=e_abs, audio=a[:, t])     # [B,Q_ev,D]

            # K/V: obj_query のみ（LNのみ適用）
            kv_feats, kv_mask, idxs = self.build_kv(obj_query, t)  # [B,F*Q_obj,D], [B,F*Q_obj]

            # L 層更新
            q = E_t0
            delta_sec = torch.tensor([(tau - t)/self.fps for tau in idxs], device=obj_query.device)  # [F]
            for layer in self.layers:
                q = layer(q, kv_feats, kv_mask, self.rel_pe, self.rel_proj, delta_sec,
                          Q_obj=Q_obj, topk_tau=self.topk_tau, topk_slot=self.topk_slot)
            outs.append(q.unsqueeze(1))
            E_prev = q
        return torch.cat(outs, dim=1)  # [B,T,Q_ev,D]

    def build_kv(self, obj_query, t_center):
        B,T,Q_obj,D = obj_query.shape
        idxs = [tau for tau in range(t_center-self.k, t_center+self.k+1)]
        kv_list, mask_list = [] , []
        for tau in idxs:
            if 0 <= tau < T:
                kv_list.append(F.layer_norm(obj_query[:, tau], (D,)))      # [B,Q_obj,D]
                mask_list.append(torch.zeros(B, Q_obj, dtype=torch.bool, device=obj_query.device))
            else:
                kv_list.append(torch.zeros(B, Q_obj, D, device=obj_query.device))
                mask_list.append(torch.ones(B, Q_obj, dtype=torch.bool, device=obj_query.device))
        kv_feats = torch.cat(kv_list, dim=1)   # [B,F*Q_obj,D]
        kv_mask  = torch.cat(mask_list, dim=1) # [B,F*Q_obj]
        return kv_feats, kv_mask, idxs
```

### 3.4 クエリ初期化（凸結合）

```python
class EventQueryMergeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_prev = nn.Linear(dim, 1)
        self.w_init = nn.Linear(dim, 1)
        self.ctx_proj  = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)
        self.audio_proj= nn.Linear(dim, dim)
        self.out_norm  = nn.LayerNorm(dim)
    def forward(self, E_prev, E_init, ctx, time, audio):
        # 凸結合
        a_prev = self.w_prev(E_prev); a_init = self.w_init(E_init)
        w = torch.softmax(torch.cat([a_prev, a_init], dim=-1), dim=-1)
        base = w[...,0:1]*E_prev + w[...,1:2]*E_init
        # バイアス加算（KVには入れない）
        bias = self.ctx_proj(ctx).unsqueeze(1) + self.time_proj(time).unsqueeze(0).unsqueeze(1) + self.audio_proj(audio).unsqueeze(1)
        return self.out_norm(base + bias)
```

### 3.5 Deformable 層（slot×time、離散時間）

* 目的: 各クエリ `q` が、時間窓内 `idxs` の**近傍フレーム `topk_tau`** と、各フレームの**近傍スロット `topk_slot`** のみを対象に、`M` 点の**slotサンプル**を生成して疎に集約する。
* 時間は**離散**（各 τ のトークンからサンプル）。`RelTimePE(Δ)` は**重みロジット生成**の特徴として使用し、`−λ|Δ|` を減算。

```python
class EventDeformableDecoderLayer(nn.Module):
    def __init__(self, dim, num_points, offset_mode="per_tau"):
        super().__init__()
        self.dim = dim; self.M = num_points; self.mode = offset_mode
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.ref_slot = nn.Linear(dim, 1)     # 参照スロット中心（連続→後で近傍離散に）
        self.delta_mlp = nn.Linear(2*dim, self.M)   # slot方向のサンプルロジット（近傍候補内でsoftmax）
        self.alpha_mlp = nn.Linear(2*dim, self.M)   # サンプル重みロジット
        self.lam = nn.Parameter(torch.tensor(1.0))  # 距離ペナルティ係数
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, q, kv_feats, kv_mask, rel_pe, rel_proj, delta_sec, Q_obj, topk_tau, topk_slot):
        # q: [B,Q_ev,D]; kv_feats: [B,F*Q_obj,D]; kv_mask: [B,F*Q_obj]; delta_sec: [F]
        B,Qe,D = q.shape; F = delta_sec.shape[0]
        qn = self.norm1(q)
        # 近傍フレーム選択（|Δ| 小さい順に topk_tau）
        tau_idx = torch.topk(-delta_sec.abs(), k=topk_tau).indices.sort()[0]  # [topk_tau]
        # 参照スロット中心（連続 0..Q_obj-1）
        ref = torch.sigmoid(self.ref_slot(qn)) * (Q_obj - 1)  # [B,Qe,1]
        ref_round = ref.round().clamp(0, Q_obj-1)             # [B,Qe,1]

        # 集約
        agg = torch.zeros(B,Qe,D, device=kv_feats.device)
        for i, tau in enumerate(tau_idx):
            # τ ブロックの切り出し
            start = tau*Q_obj; end = start+Q_obj
            kv_tau = kv_feats[:, start:end, :]    # [B,Q_obj,D]
            mask_tau = kv_mask[:, start:end]      # [B,Q_obj]

            # 相対時刻特徴 + クエリ特徴
            rel_feat = rel_proj(rel_pe(delta_sec[tau].view(1))).view(1,1,D).expand(B,Qe,D)
            psi = torch.cat([qn, rel_feat], dim=-1)  # [B,Qe,2D]

            # 近傍スロット候補（整数インデックス）
            # ref_round ± r の範囲から topk_slot を選出（境界クランプ）
            r = torch.arange(-topk_slot//2+1, topk_slot//2+1, device=kv_feats.device)
            cand = (ref_round + r.view(1,1,-1)).clamp(0, Q_obj-1).long()  # [B,Qe,topk_slot]

            # slot サンプル重み（M 点）
            slot_logits = self.delta_mlp(psi)        # [B,Qe,M]
            slot_w = torch.softmax(slot_logits, dim=-1)  # [B,Qe,M]

            # 重みロジット（距離ペナルティ含む）
            alpha = self.alpha_mlp(psi) - self.lam*delta_sec[tau].abs()  # [B,Qe,M]
            w = torch.softmax(alpha, dim=-1)                              # [B,Qe,M]

            # cand から M 個を循環的に取り出して gather（簡易）
            idx_m = torch.arange(self.M, device=kv_feats.device) % topk_slot  # [M]
            take = cand[:,:,idx_m]  # [B,Qe,M]

            # gather & 加重平均
            # kv_tau: [B,Q_obj,D] → gather で [B,Qe,M,D]
            b_idx = torch.arange(B, device=kv_feats.device)[:,None,None].expand(B,Qe,self.M)
            q_idx = torch.arange(Qe, device=kv_feats.device)[None,:,None].expand(B,Qe,self.M)
            sampled = kv_tau[b_idx, take, : ]  # [B,Qe,M,D]
            # マスク 0/1 で無効化（無効は 0 に）
            mask_s = mask_tau[b_idx, take]     # [B,Qe,M]
            sampled = sampled * (~mask_s).unsqueeze(-1)

            agg = agg + (w*slot_w).unsqueeze(-1) * sampled  # [B,Qe,M,1]*[B,Qe,M,D]→sum M
        out = q + self.out_proj(agg.sum(dim=2))
        return out + self.ffn(self.norm2(out))
```

> 注: 上記の slot 抽出は**簡易な擬似**。実装では `topk_slot` の候補から **attention による混合**を行い、`cand` の選定も learnable に置換可能。

---

## 4. ヘッド（プーリング無し）

```python
class EventHead(nn.Module):
    def __init__(self, dim, num_events):
        super().__init__()
        self.cls = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_events))
    def forward(self, event_queries):  # [B,T,Q_ev,D]
        logits = self.cls(event_queries)       # [B,T,Q_ev,E]
        prob   = torch.sigmoid(logits)
        return prob, logits
```

---

## 5. 実装要件

* **KV は obj_query のみ**。CLS や role/exist/xyz 等の付加は行わない。
* **時刻埋め込み**: `AbsTimePE` はクエリ初期化にのみ使用。`RelTimePE` は重み生成の特徴に使用。KV には入れない。
* **音声（任意）**: `EventTimeAlign` で `T` に合わせ、学習可能遅延 `β_global`（秒）で時間シフト後、**クエリ初期化へのバイアス**として加算のみ。KV/Deformable には使わない。
* **オンライン/オフライン**: 本仕様はオフライン（未来参照可）。オンライン版は `k_future=0` で派生可能。
* **出力**: `[B,T,Q_ev,E]`。フレーム単位の統合は**後段**の責務。
* **非スコープ**: 持続イベント、サブフレーム出力、事前スコア利用。

---

## 6. 主要テンソルの形状フロー

```
obj_query [B,T,Q_obj,D_s]
  └─ z_proj → z [B,T,D_h]
  └─ (audio) align→shift→proj → a [B,T,D_h] or 0
  └─ Decoder (t=0..T-1)
       ├─ AbsTimePE(t)→abs→abs_proj [D_h]
       ├─ Merge(E_prev,E_init,z_t,abs_t,a_t) → E_t^0 [B,Q_ev,D_h]
       ├─ build_kv(obj_query,[t−k..t+k]) → kv_feats [B,F*Q_obj,D_h]
       ├─ L×{Deformable(slot×time, RelTimePE(Δ), −λ|Δ|)} → E_t^L [B,Q_ev,D_h]
  └─ concat t → [B,T,Q_ev,D_h]
  └─ Head（各クエリ）→ logits/prob [B,T,Q_ev,E]
```
