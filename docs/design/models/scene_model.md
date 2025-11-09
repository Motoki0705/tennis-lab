# シーンモデル アーキテクチャ設計

本書は、単眼動画からボール・プレーヤーの高次表現と3D属性を推定するシーンモデルの**確定仕様**を示す。学習方針は別文書とし、本書では**アーキテクチャとデータフロー**に限定する。

---

## 0. 要旨

* 共有潜在次元は `D_model` に統一。
* 画像表現は ViT。**ViT の出力に `patch_pos` は含めない**。ViT 内では学習型2D絶対位置埋め込みを使用し、**デコーダ側で固定2D sin/cos座標**を生成・付与する。
* 時系列表現は TemporalEncoder（自注意+RoPE）で得る `z_t`（内容文脈）。
* 時刻埋め込みは二系統：

  * **絶対時刻** `AbsTimePE(τ)`：Query初期化とKV構築に加算。
  * **相対時刻** `RelTimePE(Δ)`：Deformable層のオフセット・重み生成に注入し、重みロジットに距離ペナルティ `-λ|Δ|` を加える。
* デコーダは **Recurrent + Temporal-Deformable**。既定は**各参照フレーム τ ごとに (Δx,Δy,Δt) を生成**。
* KV は**各フレームのパッチのみ**（CLS は含めない）。
* オフライン処理（全フレーム一括）。
* `num_queries` は探索多様性の本数として解釈する。
* 出力：`role_logits(2)`, `exist_conf(1)`, `ball_xyz(3)`, `player_xyz(3)`, `smpl(P)`。

---

## 1. 記号・形状

* バッチ: `B`、フレーム: `T`、パッチ: `Np`、クエリ: `Q`、共有次元: `D_model=D`。
* 窓半径: `k`、窓長: `F=2k+1`。フレームレート: `fps`。
* 連続時刻[s]: `τ = t / fps`、相対時刻差[s]: `Δ = (τ_τ − τ_t)`。

### 入出力

* 入力: `frames [B,T,3,H,W]`
* 出力（dict）:

  * `role_logits [B,T,Q,2]`
  * `exist_conf  [B,T,Q,1]`
  * `ball_xyz    [B,T,Q,3]`
  * `player_xyz  [B,T,Q,3]`
  * `smpl        [B,T,Q,P]`

---

## 2. コンフィグ（API）

```python
from dataclasses import dataclass

@dataclass
class SceneConfig:
    # Image / ViT
    img_size: int
    patch_size: int
    vit_depth: int
    vit_heads: int
    D_model: int

    # Temporal encoder (RoPE)
    temporal_depth: int
    temporal_heads: int
    max_T_hint: int  # メモリ計画用ヒント

    # Time PE
    fps: float
    absK: int            # AbsTimePE 周波数数 K
    abs_Thorizon: float  # Abs の低周波境界設定用秒
    relQ: int            # RelTimePE 周波数数 Q
    rel_wmin: float      # Rel 最小角周波数 [rad/s]
    rel_wmax: float      # Rel 最大角周波数 [rad/s]

    # Decoder
    num_queries: int
    decoder_layers: int
    num_points: int      # サンプル点 M
    window_k: int
    offset_mode: str = "per_tau"  # 既定: τごと / 代替: global
    tbptt_detach: bool = True

    # Prediction head
    smpl_param_dim: int
```

---

## 3. モジュール構成とデータフロー

### 3.1 全体

```python
class SceneModel(nn.Module):
    def __init__(self, cfg: SceneConfig):
        super().__init__()
        D = cfg.D_model

        self.backbone = ViTBackbone(cfg.img_size, cfg.patch_size, D, cfg.vit_depth, cfg.vit_heads)
        self.temporal = TemporalEncoderRoPE(D, cfg.temporal_depth, cfg.temporal_heads)

        # 時刻埋め込み
        self.abs_pe   = AbsTimePE(K=cfg.absK, fps=cfg.fps, Thorizon=cfg.abs_Thorizon)   # -> 2K
        self.abs_proj = nn.Linear(2*cfg.absK, D)
        self.rel_pe   = RelTimePE(Q=cfg.relQ, wmin=cfg.rel_wmin, wmax=cfg.rel_wmax)     # -> 2Q
        self.rel_proj = nn.Linear(2*cfg.relQ, D)

        self.decoder = RecurrentTemporalDeformableDecoder(
            dim=D, num_queries=cfg.num_queries, num_layers=cfg.decoder_layers,
            num_points=cfg.num_points, k=cfg.window_k,
            offset_mode=cfg.offset_mode, tbptt_detach=cfg.tbptt_detach,
            abs_pe=self.abs_pe, abs_proj=self.abs_proj,
            rel_pe=self.rel_pe, rel_proj=self.rel_proj, fps=cfg.fps,
            img_size=cfg.img_size, patch_size=cfg.patch_size  # 2D固定座標生成に使用
        )

        self.head = PredictionHead(D, cfg.smpl_param_dim)

    def forward(self, frames):  # [B,T,3,H,W]
        patch_tokens, cls_tokens = self.backbone(frames)              # [B,T,Np,D], [B,T,D]
        z = self.temporal(cls_tokens)                                 # [B,T,D]
        decoded = self.decoder(patch_tokens, z)                       # [B,T,Q,D]
        return self.head(decoded)
```

### 3.2 ViT バックボーン

* 役割：フレームごとに `patch_tokens` と `cls_tokens` を生成。
* ViT 内では学習型2D絶対位置埋め込みを使用。
* 出力は **`patch_tokens` と `cls_tokens` のみ**（`patch_pos` は出力しない）。

```python
class ViTBackbone(nn.Module):
    def __init__(self, img_size, patch_size, dim, depth, heads):
        ...  # 標準的な ViT 実装。学習型 pos_embed はバックボーン内のみ使用
    def forward(self, frames):
        # returns: patch_tokens [B,T,Np,D], cls_tokens [B,T,D]
        ...
        return patch_tokens, cls_tokens
```

### 3.3 TemporalEncoder（RoPE）

* `cls_tokens [B,T,D]` に時間自己注意（RoPE適用）→ `z [B,T,D]`。

```python
class TemporalEncoderRoPE(nn.Module):
    def __init__(self, dim, depth, heads):
        ...
    def forward(self, cls_tokens):  # [B,T,D]
        ...
        return z  # [B,T,D]
```

---

## 4. 時刻埋め込み

### 4.1 絶対時刻 `AbsTimePE(τ)`

* 入力：`τ=t/fps`（秒）。
* 出力：`[cos(s ω_k τ + φ_k), sin(s ω_k τ + φ_k)]_{k=1..K}`（`2K` 次元）。
* `ω_k` は `[2π/Thorizon, 2π·0.45·fps]` を対数等間隔で固定（非学習）。`s` と `φ_k` は学習。
* 適用：Query初期化と KV 構築に線形射影して加算。

```python
class AbsTimePE(nn.Module):
    def __init__(self, K, fps, Thorizon):
        ...
    def forward(self, t_sec: torch.Tensor) -> torch.Tensor:  # [..., 2K]
        ...
```

### 4.2 相対時刻 `RelTimePE(Δ)`

* 入力：`Δ = τ_τ − τ_t`（秒）。
* 出力：`[cos(ω̃_q Δ), sin(ω̃_q Δ)]_{q=1..Q}`（`2Q` 次元）。
* 適用：Deformable層のオフセット・重み生成に注入。さらに重みロジットに `-λ|Δ|` を減算。

```python
class RelTimePE(nn.Module):
    def __init__(self, Q, wmin, wmax):
        ...
    def forward(self, delta_sec: torch.Tensor) -> torch.Tensor:  # [..., 2Q]
        ...
```

---

## 5. デコーダ：Recurrent + Temporal-Deformable

### 5.1 役割とフロー

1. `Q_prev`（前時刻最終）と `Q_init`（学習クエリ）を**softmax凸結合**し、`z_t` と `AbsTimePE(t)` を加えて Query 初期化。
2. `build_kv` で時間窓 `[t−k, t+k]` のパッチ特徴のみを集約。各フレームのパッチに **固定2D sin/cos座標** と `AbsTimePE(τ)` を加算。CLS は含めない。
3. Temporal-Deformable を L 層適用。**既定は τごとに (Δx,Δy,Δt) を生成**し、双線形（空間）+線形（時間）補間でサンプル→集約。

### 5.2 擬似コード（要点）

```python
class RecurrentTemporalDeformableDecoder(nn.Module):
    def __init__(self, dim, num_queries, num_layers, num_points, k,
                 offset_mode, tbptt_detach, abs_pe, abs_proj, rel_pe, rel_proj, fps,
                 img_size, patch_size):
        super().__init__()
        self.D = dim; self.Q = num_queries; self.L = num_layers
        self.M = num_points; self.k = k; self.offset_mode = offset_mode
        self.tbptt = tbptt_detach; self.abs_pe = abs_pe; self.abs_proj = abs_proj
        self.rel_pe = rel_pe; self.rel_proj = rel_proj; self.fps = fps

        # 固定2D sin/cos 座標（学習しない）：[1,Np,D]
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.Np = self.grid_h * self.grid_w
        self.register_buffer("pos2d", fixed_2d_sincos(self.grid_h, self.grid_w, dim), persistent=False)

        self.init_queries = nn.Parameter(torch.randn(self.Q, self.D))
        self.ctx_proj  = nn.Linear(self.D, self.D)  # z_t
        self.merge     = QueryMergeLayer(self.D)    # softmax凸結合
        self.layers = nn.ModuleList([
            TemporalDeformableDecoderLayer(self.D, self.M, offset_mode)
            for _ in range(self.L)
        ])

    def forward(self, patch_tokens, temporal_tokens):  # [B,T,Np,D], [B,T,D]
        B,T,Np,D = patch_tokens.shape
        Q_prev = self.init_queries.unsqueeze(0).expand(B, self.Q, D)
        outs = []
        for t in range(T):
            if self.tbptt: Q_prev = Q_prev.detach()
            # Query 初期化
            z_t = self.ctx_proj(temporal_tokens[:, t])                 # [B,D]
            e_abs_t = self.abs_proj(self.abs_pe(torch.tensor(t/self.fps, device=z_t.device)))  # [D]
            Q_init = self.init_queries.unsqueeze(0).expand_as(Q_prev)
            Q_t0 = self.merge(Q_prev, Q_init, z_t, e_abs_t)            # [B,Q,D]

            # KV 構築（CLSなし）
            kv_feats, kv_meta, kv_mask, delta_sec = self.build_kv(patch_tokens, t)

            # L 層更新
            q = Q_t0
            for layer in self.layers:
                q = layer(q, kv_feats, kv_meta, kv_mask, self.rel_pe, self.rel_proj, delta_sec)
            outs.append(q.unsqueeze(1))
            Q_prev = q
        return torch.cat(outs, dim=1)  # [B,T,Q,D]

    def build_kv(self, patch_tokens, t_center):
        B,T,Np,D = patch_tokens.shape
        idxs = [tau for tau in range(t_center-self.k, t_center+self.k+1)]
        kv_list = []; mask_list = []; meta = {"grid": (self.grid_h, self.grid_w), "block": []}
        for tau in idxs:
            if 0 <= tau < T:
                x = patch_tokens[:, tau] + self.pos2d  # [B,Np,D]
                e_abs = self.abs_proj(self.abs_pe(torch.tensor(tau/self.fps, device=x.device)))  # [D]
                x = x + e_abs.unsqueeze(0).unsqueeze(1)  # [B,Np,D]
                kv_list.append(x)
                mask_list.append(torch.zeros(B, Np, dtype=torch.bool, device=x.device))
                meta["block"].append({"tau": tau})
            else:
                kv_list.append(torch.zeros(B, Np, D, device=patch_tokens.device))
                mask_list.append(torch.ones(B, Np, dtype=torch.bool, device=patch_tokens.device))
                meta["block"].append({"tau": None})
        kv_feats = torch.cat(kv_list, dim=1)  # [B, F*Np, D]
        kv_mask  = torch.cat(mask_list, dim=1) # [B, F*Np]

        # 相対秒 Δ を返す（[B,Q,F] に拡張）
        F = len(idxs)
        delta = torch.tensor([(tau - t_center)/self.fps for tau in idxs], device=patch_tokens.device)  # [F]
        delta_sec = delta.view(1,1,F).expand(B, self.Q, F)
        return kv_feats, meta, kv_mask, delta_sec
```

#### 固定2D sin/cos 生成（学習しない）

```python
def fixed_2d_sincos(H, W, D):
    # [-1,1] の格子を 2D sin/cos 展開し、最終次元 D に射影（D が奇数の場合は切り捨て）
    # ここでは擬似：実装では標準の 2D sin/cos を生成
    return torch.zeros(1, H*W, D)
```

---

## 6. Query 初期化（凸結合）

```python
class QueryMergeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_prev = nn.Linear(dim, 1)
        self.w_init = nn.Linear(dim, 1)
        self.ctx_proj  = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)
        self.out_norm  = nn.LayerNorm(dim)
    def forward(self, Q_prev, Q_init, ctx, e_abs_t):
        a_prev = self.w_prev(Q_prev)  # [B,Q,1]
        a_init = self.w_init(Q_init)  # [B,Q,1]
        w = torch.softmax(torch.cat([a_prev, a_init], dim=-1), dim=-1)
        base = w[...,0:1]*Q_prev + w[...,1:2]*Q_init
        ctx_b = self.ctx_proj(ctx).unsqueeze(1)
        time_b = self.time_proj(e_abs_t).unsqueeze(0).unsqueeze(1)
        return self.out_norm(base + ctx_b + time_b)
```

---

## 7. Temporal-Deformable 層（τごとオフセット生成が既定）

### 7.1 仕様

* 入力クエリ `q [B,Q,D]` と時間窓参照 `τ=1..F` に対し、各 `τ` で `M` 個のサンプル `(Δx,Δy,Δt)` と重み `α` を生成。
* 入力特徴は `kv_feats [B,F*Np,D]`。`meta` でブロック境界 `(τ, H, W)` を把握し、双線形（空間）+線形（時間）補間でサンプル。
* 相対時刻 `Δ` の特徴 `RelTimePE(Δ)` を `rel_proj` で `D` 次元へ射影し、`LN(q)` と結合してオフセット・重みを生成。
* 重みロジットに距離ペナルティ `-λ|Δ|` を適用。

### 7.2 擬似コード

```python
class TemporalDeformableDecoderLayer(nn.Module):
    def __init__(self, dim, num_points, offset_mode="per_tau"):
        super().__init__()
        self.M = num_points; self.mode = offset_mode
        self.ref_xy = nn.Linear(dim, 2)  # (x,y)∈[0,1]^2 （per_tau に切替える場合は ψ 入力に変更）
        self.delta_mlp = nn.Linear(2*dim, 3*self.M)
        self.alpha_mlp = nn.Linear(2*dim, self.M)
        self.lam = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, q, kv_feats, kv_meta, kv_mask, rel_pe, rel_proj, delta_sec):
        B,Q,D = q.shape; F = delta_sec.shape[-1]
        qn = self.norm1(q)                             # [B,Q,D]
        rel_feat = rel_proj(rel_pe(delta_sec))         # [B,Q,F,D]
        psi = torch.cat([qn.unsqueeze(2).expand(B,Q,F,D), rel_feat], dim=-1)  # [B,Q,F,2D]

        delta = self.delta_mlp(psi).view(B,Q,F,self.M,3)     # [B,Q,F,M,3]
        logits = self.alpha_mlp(psi)                         # [B,Q,F,M]
        weights = torch.softmax(logits - self.lam*delta_sec.abs().unsqueeze(-1), dim=-1)

        ref = torch.sigmoid(self.ref_xy(qn)).unsqueeze(2).expand(B,Q,F,2)  # [B,Q,F,2]
        sampled = trilinear_sample(kv_feats, kv_meta, kv_mask, ref, delta) # [B,Q,F,M,D]
        agg = (weights.unsqueeze(-1) * sampled).sum(dim=3).sum(dim=2)      # [B,Q,D]

        out = q + self.proj(agg)
        return out + self.ffn(self.norm2(out))
```

### 7.3 三重補間 `trilinear_sample`（仕様）

* 入力：`kv_feats [B,F*Np,D]`、`meta`（各ブロックの開始オフセット・`(H,W)`）、`kv_mask [B,F*Np]`、参照点 `ref(x,y)`、オフセット `delta(Δx,Δy,Δt)`。
* 出力：`[B,Q,F,M,D]`。
* 手順：

  1. `(x+Δx, y+Δy)` を `(H,W)` グリッドへ変換し、**双線形補間**でパッチ特徴を取得。
  2. 時間軸は `τ+Δt` で **線形補間**。範囲外は端点でクランプ。
  3. `kv_mask` は無効トークンへのサンプル重みを0化。

---

## 8. 予測ヘッド

```python
class PredictionHead(nn.Module):
    def __init__(self, dim, smpl_param_dim):
        super().__init__()
        self.role  = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        self.exist = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.ball_xyz   = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        self.player_xyz = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        self.smpl       = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, smpl_param_dim))
    def forward(self, x):  # [B,T,Q,D]
        return {
            "role_logits": self.role(x),
            "exist_conf":  torch.sigmoid(self.exist(x)),
            "ball_xyz":    self.ball_xyz(x),
            "player_xyz":  self.player_xyz(x),
            "smpl":        self.smpl(x),
        }
```

---

## 9. 実装要件（抜粋）

* 入力前処理：リサイズ・中心クロップで `H=W=img_size`、標準化。
* TemporalEncoder は RoPE を Q/K に適用。絶対学習の時間埋め込みは使用しない。
* KV 構築では CLS を含めず、**固定2D座標**と **AbsTimePE(τ)** を加算する。
* Deformable 層は **相対時刻特徴**と **距離ペナルティ** を用いる。
* 再帰は既定で `detach` を行い、オフラインで全フレームを順次処理する。

---

## 10. 主要テンソルの形状フロー

```
frames [B,T,3,H,W]
  └─ ViTBackbone → patch_tokens [B,T,Np,D], cls_tokens [B,T,D]
  └─ TemporalEncoderRoPE → z [B,T,D]
  └─ Decoder (for t in 1..T)
       ├─ AbsTimePE(t) → abs_vec→abs_proj [D]
       ├─ Merge(Q_prev,Q_init,z_t,abs_t) → Q_t0 [B,Q,D]
       ├─ build_kv([t−k..t+k]): (patch + pos2d + AbsTimePE(τ)) →
       │     kv_feats [B,F*Np,D], kv_mask [B,F*Np], delta_sec [B,Q,F]
       ├─ L×{Deformable(τごとオフセット, RelTimePE(Δ), −λ|Δ|)} → q [B,Q,D]
  └─ concat t → decoded [B,T,Q,D]
  └─ Head → {role_logits, exist_conf, ball_xyz, player_xyz, smpl}
```
