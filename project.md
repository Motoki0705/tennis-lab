# プロジェクト

## プロジェクト概要
### 目的
- テニスの試合を解析するためのシステム全体を構築し、高精度で再現性の高い分析結果を安定的に提供する。
- 単眼カメラから3Dシーンを復元し、ボール・プレーヤーの位置やポーズをマルチモーダルに推論できるモデル群を開発する。
- 推論結果をもとにイベント検知と統計解析を実現し、LLMでユーザーに理解しやすいレポートへ変換する。

### 主要要素
- **モデル開発**: シーン検知モデルとイベント検知モデルを中心とした認識パイプラインを構築。
- **データ戦略**: 既存データセットと自作データセットを組み合わせ、高品質で大量なデータを柔軟に確保。
- **システム統合**: マルチカメラによる精度向上と、LLMによる考察生成を含むエンドツーエンドな解析体験を提供。

### データとLLM活用の方針
- マルチカメラ環境で取得したデータに加え、自作データセットを継続的に拡充してモデル精度を底上げする。
- 統計量の生成方法を事前に定義したうえでRAGを構成し、必要であればLLMをファインチューニングして示唆に富む解説を生成する。

## モデル構想
### 1. シーン検知モデル
#### 目的
- 単眼カメラの連続画像からボール・プレーヤーの3D位置とポーズを復元し、コートに対する相対位置を推論する（視覚特徴のみを利用）。

#### アーキテクチャ
前提:
- PyTorch風の擬似コード（必ずしも完全動作しないが、構造は忠実）
- キーアイデア:
  - ViTバックボーン
  - Temporal Encoder（時間コンテキスト）
  - 明示的な時間埋め込み
  - Recurrentなクエリ（時刻間で持ち越し）
  - `init_queries`（恒常的探索用）
  - 時系列トークン（$z_t$）と時間埋め込み（$e_t$）を分離し、クエリ初期化時に明示的に注入
  - Temporal-Deformableなデコーダ（ローカル時間窓＋疎サンプリング）
- 数式は説明内で口頭化し、コード側は直感優先

構成:
- `SceneModel`
- `ViTBackbone`
- `TemporalEncoder`
- `TimeEmbedding`
- `RecurrentTemporalDeformableDecoder`
  - `QueryMergeLayer`
  - `build_kv`（時空間特徴構築）
  - `TemporalDeformableDecoderLayer`（サンプリングの概念）
- `PredictionHead`

1. 全体: SceneModel

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneModel(nn.Module):
    """
    単眼動画から [T, Q, D_out] を推定するモデルのトップレベル。
    - 空間: ViTバックボーン
    - 時間: TemporalEncoder
    - 時刻埋め込み: TimeEmbedding
    - 検出/トラッキング: RecurrentTemporalDeformableDecoder
    - 出力: PredictionHead で3D位置・ポーズなどを回帰
    """

    def __init__(self, config):
        super().__init__()
        dim = config.dim

        self.backbone = ViTBackbone(
            img_size=config.img_size,
            patch_size=config.patch_size,
            dim=dim,
            depth=config.vit_depth,
            heads=config.vit_heads,
        )

        self.temporal_encoder = TemporalEncoder(
            dim=dim,
            depth=config.temporal_depth,
            heads=config.temporal_heads,
            use_rel_pos=True,
            max_T=config.max_T,
        )

        self.time_embed = TimeEmbedding(
            max_len=config.max_T,
            dim=dim,
            mode="sin_learnable_scale",  # 実装ポリシーは後述
        )

        self.decoder = RecurrentTemporalDeformableDecoder(
            dim=dim,
            num_queries=config.num_queries,
            num_layers=config.decoder_layers,
            num_points=config.num_points,
            k=config.window_k,
        )

        self.head = PredictionHead(
            dim=dim,
            d_out=config.d_out,  # 例: 3D位置 + ポーズ + conf + クラス 等
        )

    def forward(self, frames):
        """
        frames: [B, T, 3, H, W]
        return: [B, T, Q, D_out]
        """
        patch_tokens, cls_tokens, patch_pos = self.backbone(frames)
        # patch_tokens: [B, T, N_p, D]
        # cls_tokens:   [B, T, D]

        temporal_tokens = self.temporal_encoder(cls_tokens)
        # temporal_tokens: [B, T, D] (z_t)

        decoded = self.decoder(
            patch_tokens=patch_tokens,
            temporal_tokens=temporal_tokens,
            patch_pos=patch_pos,
            time_embed=self.time_embed,
        )
        # decoded: [B, T, Q, D]

        out = self.head(decoded)
        # out: [B, T, Q, D_out]
        return out
```

2. ViTBackbone

目的:
- 各フレームごとにパッチトークンとCLSトークンを生成
- 空間位置エンコーディングを付与
- Transformerブロック内で正規化・自己注意

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B*T, 3, H, W]
        x = self.proj(x)  # [B*T, D, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B*T, N_p, D]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # x: [B, N, D]
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class ViTBackbone(nn.Module):
    def __init__(self, img_size, patch_size, dim, depth, heads):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # CLS + patch 用の位置埋め込み
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.patch_embed.num_patches, dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, frames):
        """
        frames: [B, T, 3, H, W]
        return:
            patch_tokens: [B, T, N_p, D]
            cls_tokens:   [B, T, D]
            patch_pos:    [1, 1, N_p, D]
        """
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)

        x = self.patch_embed(x)  # [B*T, N_p, D]
        B_T, N_p, D = x.shape

        cls_tokens = self.cls_token.expand(B_T, -1, -1)  # [B*T, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)            # [B*T, 1+N_p, D]
        x = x + self.pos_embed[:, :1+N_p, :]             # 位置埋め込み加算

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls = x[:, 0]          # [B*T, D]
        patch = x[:, 1:]       # [B*T, N_p, D]

        cls = cls.view(B, T, D)
        patch = patch.view(B, T, N_p, D)

        patch_pos = self.pos_embed[:, 1:1+N_p, :].unsqueeze(1)  # [1,1,N_p,D]

        return patch, cls, patch_pos
```

3. TemporalEncoder

目的:
- CLSトークン列に時間方向の自己注意をかけ、時間文脈付き時系列トークン $z_t$ を得る。
- 絶対位置埋め込み＋相対位置バイアスを利用可能。
- 出力は [B, T, D]。

```python
class TemporalEncoder(nn.Module):
    def __init__(self, dim, depth, heads, use_rel_pos=True, max_T=1024):
        super().__init__()
        self.use_rel_pos = use_rel_pos
        self.abs_pos = nn.Parameter(torch.randn(1, max_T, dim))

        self.blocks = nn.ModuleList([
            TemporalBlock(dim, heads, use_rel_pos=use_rel_pos, max_T=max_T)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, cls_tokens):
        # cls_tokens: [B, T, D]
        B, T, D = cls_tokens.shape
        x = cls_tokens + self.abs_pos[:, :T, :]  # 絶対時間埋め込み

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x  # [B, T, D]


class TemporalBlock(nn.Module):
    def __init__(self, dim, heads, use_rel_pos=True, max_T=1024):
        super().__init__()
        self.use_rel_pos = use_rel_pos
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        )
        if use_rel_pos:
            # 相対位置バイアス: 簡略モデル
            self.rel_bias = nn.Parameter(
                torch.zeros(2*max_T-1)
            )  # 実装では [heads, 2*max_T-1] 等にする

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape

        h = self.norm1(x)
        # 通常のMultiheadAttention呼び出しにrel biasを加えたいが、
        # ここでは概念のみ示し、実装は省略。
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x
```

4. TimeEmbedding

目的:
- 純粋な時刻情報 $e_t$ を提供（z_tとは分離）
- sin/cos＋学習スケール、または完全学習埋め込みなど

```python
class TimeEmbedding(nn.Module):
    def __init__(self, max_len, dim, mode="sin_learnable_scale"):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        self.mode = mode

        if mode == "learned":
            self.emb = nn.Embedding(max_len, dim)
        elif mode == "sin_learnable_scale":
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            raise NotImplementedError

    def forward(self, t):
        """
        t: int または tensor of shape [B] でも可（ここではint想定）
        return: [1, D] or [B, D]
        """
        if self.mode == "learned":
            idx = torch.tensor([t], dtype=torch.long, device=self.emb.weight.device)
            return self.emb(idx)  # [1, D]

        if self.mode == "sin_learnable_scale":
            pos = torch.tensor([t], dtype=torch.float32, device=self.scale.device)
            dim = self.dim
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=pos.device) / dim))
            sinusoid_inp = self.scale * pos.unsqueeze(-1) * inv_freq
            sin = torch.sin(sinusoid_inp)
            cos = torch.cos(sinusoid_inp)
            pe = torch.zeros(1, dim, device=pos.device)
            pe[0, 0::2] = sin
            pe[0, 1::2] = cos
            return pe  # [1, D]
```

5. RecurrentTemporalDeformableDecoder

目的:
- 時刻を跨いでクエリを持ち越す
- 各時刻で:
  - 前時刻クエリ $Q_{t-1}^{(L)}$
  - init_queries（探索用）
  - 時系列トークン z_t（文脈）
  - 時刻埋め込み e_t（時間情報）
  を `QueryMergeLayer`で統合し、$Q_t^{(0)}$ を生成
- その後、ローカル時間窓から構成したKVに対してDeformable的注意で更新

```python
class RecurrentTemporalDeformableDecoder(nn.Module):
    def __init__(self, dim, num_queries, num_layers, num_points, k):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_points = num_points
        self.k = k  # 時間ウィンドウ半径

        # 恒常的な初期クエリ (init Q)
        self.init_queries = nn.Parameter(torch.randn(num_queries, dim))

        # z_t -> context
        self.ctx_proj = nn.Linear(dim, dim)
        # time_embed -> time成分（別途TimeEmbeddingで生成）
        self.time_proj = nn.Linear(dim, dim)

        self.merge = QueryMergeLayer(dim)

        self.layers = nn.ModuleList([
            TemporalDeformableDecoderLayer(dim, num_points)
            for _ in range(num_layers)
        ])

    def forward(self, patch_tokens, temporal_tokens, patch_pos, time_embed):
        """
        patch_tokens:   [B, T, N_p, D]
        temporal_tokens:[B, T, D]
        patch_pos:      [1, 1, N_p, D]
        time_embed:     TimeEmbedding module

        return:
            decoded: [B, T, Q, D]
        """
        B, T, N_p, D = patch_tokens.shape
        Q = self.num_queries

        # t=0: init_queries をベースに
        Q_prev = self.init_queries.unsqueeze(0).expand(B, Q, D)  # [B, Q, D]

        outputs = []

        for t in range(T):
            # 1. 時刻tの文脈と時間埋め込み
            z_t = temporal_tokens[:, t]       # [B, D]
            ctx_t = self.ctx_proj(z_t)        # [B, D]

            e_t = time_embed(t)               # [1, D]
            e_t = self.time_proj(e_t)         # [1, D]
            e_t = e_t.expand(B, -1)           # [B, D]

            # 2. init_queries の複製
            Q_init = self.init_queries.unsqueeze(0).expand(B, Q, D)  # [B, Q, D]

            # 3. QueryMergeLayer で Q_t^(0) を生成
            Q_t0 = self.merge(
                Q_prev=Q_prev,
                Q_init=Q_init,
                ctx=ctx_t,
                t_embed=e_t,
            )  # [B, Q, D]

            # 4. 時刻t周辺のKVを構築
            kv_feats, kv_meta = self.build_kv(
                patch_tokens, temporal_tokens, patch_pos, time_embed, t
            )
            # kv_feats: [B, L, D], L = (実際の有効フレーム数)*(N_p+1)

            # 5. Deformable風レイヤで更新
            q = Q_t0
            for layer in self.layers:
                q = layer(q, kv_feats, kv_meta)  # [B, Q, D]

            outputs.append(q.unsqueeze(1))  # [B,1,Q,D]
            Q_prev = q  # 次の時刻へ持ち越し（detachは好みで調整）

        decoded = torch.cat(outputs, dim=1)  # [B, T, Q, D]
        return decoded

    def build_kv(self, patch_tokens, temporal_tokens, patch_pos, time_embed, t_center):
        """
        時刻 t_center を中心とした [t_center-k, ..., t_center+k] から
        - temporal_tokens (z_tau)
        - patch_tokens
        - patch_pos
        - time_embed
        を組み合わせて K/V を構築。

        各 tau について:
        - z_tau: [B, D] -> [B,1,D]
        - x_tau: [B, N_p, D] + patch_pos
        - t_embed(tau) を両者に加算 (時間情報)
        """
        B, T, N_p, D = patch_tokens.shape
        indices = []
        for dt in range(-self.k, self.k + 1):
            tau = t_center + dt
            if 0 <= tau < T:
                indices.append(tau)

        kv_list = []
        meta = []  # 実装では各トークンの (tau, patch_index) を格納

        for tau in indices:
            z_tau = temporal_tokens[:, tau]          # [B, D]
            z_tau = z_tau.unsqueeze(1)               # [B, 1, D]

            x_tau = patch_tokens[:, tau] + patch_pos # [B, N_p, D]

            te_tau = time_embed(tau)                 # [1, D]
            te_tau = te_tau.expand(B, -1)            # [B, D]

            z_tau = z_tau + te_tau.unsqueeze(1)      # [B,1,D]
            x_tau = x_tau + te_tau.unsqueeze(1)      # [B,N_p,D]

            kv = torch.cat([z_tau, x_tau], dim=1)    # [B, N_p+1, D]
            kv_list.append(kv)

            # metaには「このブロックに対応するtau」などを入れておく
            meta.append({"time_index": tau})

        kv_feats = torch.cat(kv_list, dim=1)  # [B, L, D]
        return kv_feats, meta
```

6. QueryMergeLayer

目的:
- 前時刻クエリ Q_prev
- init_queries Q_init
- ctx_t（z_t由来の文脈）
- t_embed（時刻情報）
を統合して Q_t^(0) を作る。

挙動:
- クエリごとに、prev vs init のゲート
- 全クエリ共通で文脈と時間を加算
- LayerNormで正規化

```python
class QueryMergeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_prev = nn.Linear(dim, dim)
        self.gate_init = nn.Linear(dim, dim)
        self.ctx_proj = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, Q_prev, Q_init, ctx, t_embed):
        """
        Q_prev: [B, Q, D]
        Q_init: [B, Q, D]
        ctx:    [B, D]
        t_embed:[B, D]
        """
        B, Q, D = Q_prev.shape

        gate_prev = torch.sigmoid(self.gate_prev(Q_prev))  # [B, Q, D]
        gate_init = torch.sigmoid(self.gate_init(Q_init))  # [B, Q, D]

        base = gate_prev * Q_prev + gate_init * Q_init     # [B, Q, D]

        # 文脈と時間は全クエリに共有注入
        ctx_b = self.ctx_proj(ctx).unsqueeze(1)            # [B,1,D]
        time_b = self.time_proj(t_embed).unsqueeze(1)      # [B,1,D]

        Q_t0 = base + ctx_b + time_b                       # [B, Q, D]
        Q_t0 = self.out_norm(Q_t0)
        return Q_t0
```

7. TemporalDeformableDecoderLayer（概念）

目的:
- 各クエリから「重要そうな時空間位置」を少数サンプルし、その特徴を集約してクエリ更新。
- 本質的には MSDeformableAttention の time×space版。
- ここでは概念を示す（実運用では公式実装に近い形にする）。

```python
class TemporalDeformableDecoderLayer(nn.Module):
    def __init__(self, dim, num_points):
        super().__init__()
        self.dim = dim
        self.num_points = num_points

        # クエリからサンプリングオフセットを生成
        # ここでは簡略化して [Δindex] をL方向に出すイメージ
        self.offset_mlp = nn.Linear(dim, num_points)

        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        )

    def forward(self, queries, kv_feats, kv_meta):
        """
        queries:  [B, Q, D]
        kv_feats: [B, L, D]
        kv_meta:  メタ情報（ここでは未使用、実装では座標対応に必須）

        戻り値:
            [B, Q, D]
        """
        B, Q, D = queries.shape
        _, L, _ = kv_feats.shape

        # 1. 正規化
        q_norm = self.norm1(queries)

        # 2. サンプリングオフセット生成
        offsets = self.offset_mlp(q_norm)  # [B, Q, num_points]
        # 実装では tanhやスケーリングを入れ、base index + offset -> サンプル位置を決定
        # ここでは簡略化して softmax で重みとして扱う（密注意の疎近似イメージ）
        weights = F.softmax(offsets, dim=-1)  # [B, Q, num_points]

        # 3. （概念的）kv_featsから num_points 個サンプルを取る
        # 実際には offsets を index/座標に変換し、その位置の特徴を補間取得する。
        # ここでは単純化: ランダム/固定のインデックスを使う例（疑似）。
        # ※設計意図: 少数点への疎なアクセスで時間×空間特徴を読む。
        idx = torch.randint(low=0, high=L, size=(self.num_points,), device=kv_feats.device)
        sampled = kv_feats[:, idx, :]  # [B, num_points, D]
        # 重みをかけて集約（超簡略化）
        # broadcast: [B,Q,num_points,1] * [B,1,num_points,D]
        sampled = (weights.unsqueeze(-1) * sampled.unsqueeze(1)).sum(dim=2)  # [B,Q,D]

        # 4. 残差接続でクエリ更新
        updated = queries + self.proj(sampled)

        # 5. FFN + 残差
        h = self.norm2(updated)
        h = self.ffn(h)
        updated = updated + h

        return updated
```

実際のDeformable Attentionでは:
- kv_metaに (time_index, patch_row, patch_col) を持たせる
- クエリごとに base reference (e.g., 前フレームの予測位置を正規化座標で) を持つ
- offset_mlpから (Δt, Δx, Δy) を出し、連続座標 -> 近傍トークン補間
- この構造で time×space 上の疎注意を実現する

8. PredictionHead

目的:
- 各クエリ表現から、3D位置・ポーズ・信頼度などを回帰。
- LayerNorm＋MLPで安定化。

```python
class PredictionHead(nn.Module):
    def __init__(self, dim, d_out):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, d_out),
        )

    def forward(self, x):
        # x: [B, T, Q, D]
        x = self.norm(x)
        out = self.mlp(x)  # [B, T, Q, D_out]
        return out
```

9. 全体設計の要点（コードで反映された思想）

- 時間構造:
  - TemporalEncoderでz_tに時間文脈を埋め込み
  - TimeEmbeddingで純粋な時刻情報を別チャネルで提供
- クエリ更新:
  - Q_prevを次フレームへ持ち越し（recurrent）
  - init_queriesを常に混ぜ、「新たな物体を探すモード」を維持
  - ctx(z_t)とtime_embed(e_t)を全クエリに加え、そのフレームの文脈と時刻を共有
- 時空間情報取得:
  - build_kvでローカル時間窓 [t-k, t+k] のz_tauとpatchを統合
  - TemporalDeformableDecoderLayerで疎サンプリングにより重要点から情報取得
- 正規化:
  - 各ブロックにLayerNorm
  - QueryMergeLayer出力にもLayerNorm
  - これにより、深い再帰・多層構造でも数値安定

#### 数式表現
0. 記法と前提 

* 入力: 単眼動画
  [
  \mathcal{X}={x_t}_{t=1}^T,\quad x_t\in\mathbb{R}^{H\times W\times 3}
  ]
* 画像パッチ分割（パッチ数 (N_p)）。位置埋め込み (\pi^{\text{sp}}\in\mathbb{R}^{N_p\times D})。
* 時刻埋め込み（学習スケール付き正弦）
  [
  e_t = \text{Sinusoid}(t;,s)\in\mathbb{R}^{D},\quad s\in\mathbb{R}_{>0}\ \text{(学習パラメータ)}
  ]
* 以降、特徴次元はすべて (D)。層正規化・残差は明示した箇所以外も付随するとします。

---

1. シーンモデル（3D位置・ポーズ・トラッキング一体）

1.1 空間エンコード（ViT）

各フレーム (t) をパッチ列へ投影:
[
P_t={p_{t,n}}_{n=1}^{N_p}\in\mathbb{R}^{N_p\times D},\quad
\tilde{P}_t=\big[\mathrm{CLS}*t;;P_t\big]+\big[\pi^{\text{cls}};;\pi^{\text{sp}}\big]
]
Transformer スタック (\mathcal{T}*{\text{vit}}) により
[
[\hat{c}_t;;\hat{P}*t]=\mathcal{T}*{\text{vit}}(\tilde{P}_t),\quad
\hat{c}_t\in\mathbb{R}^{D},;\hat{P}_t\in\mathbb{R}^{N_p\times D}
]

1.2 時間エンコード（CLS系列）

絶対時間埋め込みを加え、時系列自己注意で文脈化:
[
z_t=\mathcal{T}*{\text{temp}}\big(\hat{c}*{1:T}+E^{\text{abs}}_{1:T}\big)*t\in\mathbb{R}^{D}
]
ここで (E^{\text{abs}}*{t}\in\mathbb{R}^{D}) は学習可能な絶対時刻埋め込み。

1.3 時刻付き K/V 構築（ローカル時間窓）

中心時刻 (t) に対し、半径 (k) の窓 (\mathcal{W}(t)={\tau\mid|,\tau-t,|\le k,\ 1\le\tau\le T})。
各 (\tau\in\mathcal{W}(t)) について
[
\underbrace{\bar{c}*{\tau}}*{1\times D}=\hat{c}*{\tau}+e*\tau,\quad
\underbrace{\bar{P}*{\tau}}*{N_p\times D}=\hat{P}*{\tau}+\pi^{\text{sp}}+e*\tau
]
K/V 行列は
[
\mathbf{K}^{(t)}=\mathbf{V}^{(t)}=\big[\bar{c}*{\tau};\ \bar{P}*{\tau}\big]_{\tau\in\mathcal{W}(t)}\in\mathbb{R}^{L\times D},\quad
L=|\mathcal{W}(t)|(N_p+1)
]

1.4 再帰クエリ初期化（探索＋持ち越し＋時間・文脈注入）

クエリ数 (Q)。学習可能な恒常クエリ (\mathbf{Q}^{\text{init}}\in\mathbb{R}^{Q\times D})。
前時刻最終クエリ (\mathbf{Q}^{L}*{t-1}) を用い、初期状態を融合:
[
\begin{aligned}
\mathbf{G}*{\text{prev}} &= \sigma(\mathbf{W}*{\text{gp}}\mathbf{Q}^{L}*{t-1}),\quad
\mathbf{G}*{\text{init}} = \sigma(\mathbf{W}*{\text{gi}}\mathbf{Q}^{\text{init}}) \
\mathbf{B}*t &= \mathbf{G}*{\text{prev}}\odot \mathbf{Q}^{L}*{t-1} + \mathbf{G}*{\text{init}}\odot \mathbf{Q}^{\text{init}} \
\mathbf{Q}^{0}_t &= \mathrm{LN}\Big(\mathbf{B}_t + \mathbf{1}_Q z_t^\top \mathbf{W}_z + \mathbf{1}_Q e_t^\top \mathbf{W}_e\Big)
\end{aligned}
]
（(\mathbf{1}_Q) は (Q\times 1) の全 1 ベクトル）

1.5 Temporal-Deformable デコード（疎サンプリング注意）

各層 (\ell=1,\dots,L) で、クエリごとに (M) 点を連続座標でサンプル。
クエリ (\mathbf{q}\in\mathbb{R}^{D}) から、基準参照 (\mathbf{r}(\mathbf{q})=(u,v,\tau)) とオフセット ({\Delta \mathbf{r}*m(\mathbf{q})}*{m=1}^M) を生成（学習 MLP）。
連続座標 (\mathbf{r}_m!=!\mathbf{r}(\mathbf{q})+\Delta \mathbf{r}_m) に対し、(\mathbf{V}^{(t)}) 上で双線形/三線形補間 (\Phi) を用いて値を取得:
[
\mathbf{v}_m=\Phi\big(\mathbf{V}^{(t)},\mathbf{r}_m\big)\in\mathbb{R}^{D}
]
重み (\alpha_m=\text{softmax}*m(w_m(\mathbf{q})))（学習 MLP）で集約:
[
\mathrm{DefAtt}(\mathbf{q},\mathbf{V}^{(t)})=\sum*{m=1}^M \alpha_m,\mathbf{v}_m
]
各層の更新は
[
\mathbf{Q}^{\ell}_t=\mathrm{FFN}\Big(\mathbf{Q}^{\ell-1}_t+\mathbf{W}_o\cdot
\mathrm{DefAtt}\big(\mathbf{Q}^{\ell-1}_t,\mathbf{V}^{(t)}\big)\Big)
]
最終 (\mathbf{Q}^{L}_t\in\mathbb{R}^{Q\times D}) を次時刻に持ち越す。

1.6 予測ヘッド（3D位置・ポーズ・属性）

各クエリ (j) から
[
\mathbf{y}*{t,j}=h(\mathbf{Q}^{L}*{t,j})\in\mathbb{R}^{D_{\text{out}}}
]
例：(\mathbf{y}*{t,j}=[\mathbf{p}^{3\mathrm{D}}*{t,j},\ \boldsymbol{\theta}*{t,j},\ s*{t,j},\ \text{class}_{t,j}])
（3D位置、ポーズパラメータ、信頼度、クラス等）

#### 実験ロードマップ
- 新規アーキテクチャの挙動が未確定なため、まずは単一モーダルの小規模データセットで動作検証を行う。
- 精度・安定性が確認でき次第、マルチモーダル入力と大規模データへ段階的に拡張する。

### 2. イベント検知モデル
#### 目的
- シーン検知結果を基に、事前定義したイベントの発生確率をフレーム単位で推定する。
#### アーキテクチャ
前提コンセプト
- シーンモデル:
  - 動画からオブジェクト中心の高次表現を生成する。
  - 特に:
    - `obj_query[t, j]`:
      - 各時刻$t$・各クエリ$j$についての高レベルオブジェクト表現。
      - すでに空間パッチ・時間文脈・トラッキング情報が統合された「結論的視覚コンテキスト」。
    - `temporal_tokens[t] = z_t`:
      - フレーム$t$に対するグローバル時間文脈（CLS系列をTemporalEncoderに通したもの）。
- イベントモデル:
  - 主役はシーンモデルの `obj_query`。
  - イベントクエリは `obj_query` 空間上を探索して「どのオブジェクト・どの時刻まわりが、どのイベントに相当するか」を推論する。
  - `temporal_tokens (z_t)` と `audio_tokens` は:
    - K/Vには含めない。
    - あくまで「イベントクエリ側へのバイアス・条件情報」としてのみ使用。
  - イベントクエリは時間方向にrecurrent:
    - $E_t^{(0)}$ 生成時に $E_{t-1}^{(L)}$ を明示的に取り込む。
  - K/V:
    - 時間窓 $[t-k, t+k]$ の `obj_query` のみ。
    - Deformable Attentionで疎に参照。

目的:
- 出力 $[B, T, E]$:
  - 各フレーム$t$について各イベント種の発生確率を返す。

構成モジュール

- `EventModel`
- `EventTimeAlign`
- `EventQueryDecoder`
  - `EventQueryMergeLayer`
  - `build_kv`（KV = 近傍時刻のobj_queryのみ）
  - `EventDeformableDecoderLayer`
- `EventHead`

以下、PyTorch風の擬似コードで一気に示します。

1. EventModel（トップレベル）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class EventModel(nn.Module):
    """
    シーンモデルの出力 (obj_query, temporal_tokens) と音声系列から、
    フレーム単位のイベント発生確率 [B, T, E] を推定するモデル。

    ポイント:
    - KVはシーンモデルdecoderの obj_query のみ。
      -> イベント検知は高次オブジェクト表現空間上で行う。
    - temporal_tokens(z_t) と audio はイベントクエリ生成のバイアスとしてのみ使用。
    - イベントクエリは時間方向にrecurrentで、t-1のクエリをtへ引き継ぐ。
    - 各フレーム t のイベントクエリは、時間窓 [t-k, t+k] の obj_query を
      Deformable Attentionで疎に参照し更新される。
    """

    def __init__(self, config):
        super().__init__()

        self.D_s = config.D_s          # シーンモデルの特徴次元 (obj_query, temporal_tokens)
        self.D_a = config.D_a          # 音声特徴次元 (audio_tokens)
        self.D_h = config.D_h          # イベントクエリ空間次元 (D_s と同一 or 近傍を推奨)
        self.E   = config.num_events

        # 映像Tと音声T_aの同期
        self.time_align = EventTimeAlign(mode="interp")

        # temporal_tokens(z_t) -> D_h
        self.temp_proj = nn.Linear(self.D_s, self.D_h)

        # audio_tokens -> D_h
        self.audio_proj = nn.Linear(self.D_a, self.D_h)

        # イベントクエリデコーダ (DETR形式, recurrent, deformable attention)
        self.decoder = EventQueryDecoder(
            dim=self.D_h,
            num_event_queries=config.num_event_queries,
            num_layers=config.ev_decoder_layers,
            num_points=config.ev_num_points,
            window_k=config.window_k,
        )

        # イベント分類ヘッド: イベントクエリ列 -> [B, T, E]
        self.head = EventHead(
            dim=self.D_h,
            num_events=self.E,
            pool_type="mean",  # Q_ev 次元上の集約方法
        )

    def forward(self, obj_query, temporal_tokens, audio_tokens):
        """
        obj_query:       [B, T, Q_obj, D_s]
            - シーンモデルdecoder出力クエリ (物体/プレーヤ/ボール/役割などの高次表現)
        temporal_tokens: [B, T, D_s]
            - シーンモデルのTemporalEncoder出力 z_t (グローバル時間文脈)
        audio_tokens:    [B, T_a, D_a]
            - Encodec等から得た音声潜在系列
        """
        B, T, Q_obj, D_s = obj_query.shape

        # 1. 時間文脈と音声をイベント空間次元へ
        z = self.temp_proj(temporal_tokens)             # [B, T, D_h]

        aligned_audio = self.time_align(audio_tokens, T)  # [B, T, D_a]
        a = self.audio_proj(aligned_audio)                # [B, T, D_h]

        # 2. イベントクエリデコーダ
        #    - KV: 近傍時刻の obj_query のみ
        #    - クエリ初期化: E_{t-1} + E_init + z_t + a_t
        event_queries = self.decoder(
            obj_query=obj_query,
            ctx_tokens=z,
            audio_tokens=a,
        )
        # event_queries: [B, T, Q_ev, D_h]

        # 3. イベント確率出力
        event_prob, event_logits = self.head(event_queries)
        # event_prob:   [B, T, E]
        # event_logits: [B, T, E]

        return event_prob, event_logits
```

2. EventTimeAlign（音声の時間同期）

```python
class EventTimeAlign(nn.Module):
    """
    audio_tokens [B, T_a, D_a] を映像フレーム数 T に揃える。
    シンプルな線形補間ベース。
    """

    def __init__(self, mode="interp"):
        super().__init__()
        self.mode = mode

    def forward(self, audio_tokens, T_video):
        """
        audio_tokens: [B, T_a, D_a]
        return:
            aligned: [B, T_video, D_a]
        """
        B, T_a, D_a = audio_tokens.shape

        if T_a == T_video:
            return audio_tokens

        if self.mode == "interp":
            device = audio_tokens.device
            idx = torch.linspace(0, T_a - 1, steps=T_video, device=device)
            idx0 = idx.floor().long().clamp(max=T_a - 1)
            idx1 = (idx0 + 1).clamp(max=T_a - 1)
            w = (idx - idx0.float()).unsqueeze(-1)  # [T_video,1]

            x0 = audio_tokens[:, idx0, :]  # [B, T_video, D_a]
            x1 = audio_tokens[:, idx1, :]  # [B, T_video, D_a]
            aligned = (1.0 - w) * x0 + w * x1
            return aligned

        # 他方式は必要に応じて拡張
        return audio_tokens[:, :T_video, :]
```

3. EventQueryDecoder

役割:
- 学習可能な `init_event_queries` を持つ。
- 時刻ごとに:
  - 前時刻のイベントクエリ `E_prev`
  - 初期クエリ `E_init`
  - 文脈 `z_t`
  - 音声 `a_t`
  を `EventQueryMergeLayer` で融合して `E_t^(0)` を生成。
- その後、`E_t^(0)` をクエリとして、時間窓 `[t-k, t+k]` の `obj_query` を K/V にした Deformable層で更新。
- 最終 `E_t^(L)` を次時刻に渡す。

```python
class EventQueryDecoder(nn.Module):
    def __init__(self, dim, num_event_queries, num_layers, num_points, window_k):
        super().__init__()
        self.dim = dim
        self.num_event_queries = num_event_queries
        self.num_layers = num_layers
        self.num_points = num_points
        self.k = window_k

        # 学習可能な初期イベントクエリ
        self.init_event_queries = nn.Parameter(
            torch.randn(num_event_queries, dim)
        )

        # クエリ初期化マージ層
        self.merge = EventQueryMergeLayer(dim)

        # Deformableデコーダ層
        self.layers = nn.ModuleList([
            EventDeformableDecoderLayer(dim, num_points)
            for _ in range(num_layers)
        ])

        # obj_query (D_s) -> D_h
        # ここでは dim == D_s を想定しているが、違う場合のために用意
        self.obj_proj = nn.Linear(dim, dim)

    def forward(self, obj_query, ctx_tokens, audio_tokens):
        """
        obj_query:   [B, T, Q_obj, D_s]
        ctx_tokens:  [B, T, D_h]   (z_t projected)
        audio_tokens:[B, T, D_h]   (audio projected)
        """
        B, T, Q_obj, D_s = obj_query.shape
        Q_ev = self.num_event_queries
        D_h = self.dim

        # K/V用に obj_query を投影
        obj_kv = self.obj_proj(obj_query)  # [B, T, Q_obj, D_h]

        # t=0 のイベントクエリ初期状態
        E_prev = self.init_event_queries.unsqueeze(0).expand(B, Q_ev, D_h)

        outputs = []

        for t in range(T):
            z_t = ctx_tokens[:, t]      # [B, D_h]
            a_t = audio_tokens[:, t]    # [B, D_h]

            E_init = self.init_event_queries.unsqueeze(0).expand(B, Q_ev, D_h)

            # t時刻の初期クエリ E_t^(0):
            # - E_prev (t-1の状態)
            # - E_init (恒常的なイベント探索)
            # - z_t, a_t (現在の時間コンテキスト + 音声)
            E_t0 = self.merge(
                E_prev=E_prev,
                E_init=E_init,
                ctx=z_t,
                audio=a_t,
            )  # [B, Q_ev, D_h]

            # 近傍フレームのobj_queryのみからKVを構築
            kv_feats, kv_meta = self.build_kv(obj_kv, t)

            # Deformableデコーダ層で更新
            q = E_t0
            for layer in self.layers:
                q = layer(q, kv_feats, kv_meta)  # [B, Q_ev, D_h]

            outputs.append(q.unsqueeze(1))  # [B,1,Q_ev,D_h]

            # 次時刻への持ち越し
            E_prev = q

        event_queries = torch.cat(outputs, dim=1)  # [B, T, Q_ev, D_h]
        return event_queries

    def build_kv(self, obj_kv, t_center):
        """
        KV = [t_center-k, ..., t_center+k] の obj_query のみ。
        """
        B, T, Q_obj, D_h = obj_kv.shape
        k = self.k

        indices = [
            tau for tau in range(t_center - k, t_center + k + 1)
            if 0 <= tau < T
        ]

        kv_list = []
        for tau in indices:
            kv_list.append(obj_kv[:, tau])  # [B, Q_obj, D_h]

        kv_feats = torch.cat(kv_list, dim=1)  # [B, L, D_h], L = Q_obj * len(indices)
        kv_meta = None  # 実装で座標/時刻情報を使う場合に拡張
        return kv_feats, kv_meta
```

4. EventQueryMergeLayer

- $E_{t-1}$ を明示的に取り込む。
- `E_init`（恒常探索用）、`ctx`（z_t）、`audio`（a_t）を加算。
- ゲートとLayerNormで安定化。

```python
class EventQueryMergeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_prev = nn.Linear(dim, dim)
        self.gate_init = nn.Linear(dim, dim)
        self.ctx_proj = nn.Linear(dim, dim)
        self.audio_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, E_prev, E_init, ctx, audio):
        """
        E_prev: [B, Q_ev, D]  (t-1 の最終イベントクエリ)
        E_init: [B, Q_ev, D]  (学習可能初期クエリ)
        ctx:    [B, D]        (temporal token at t, projected)
        audio:  [B, D]        (audio feature at t, projected)
        """
        B, Q_ev, D = E_prev.shape

        g_prev = torch.sigmoid(self.gate_prev(E_prev))
        g_init = torch.sigmoid(self.gate_init(E_init))

        base = g_prev * E_prev + g_init * E_init  # [B, Q_ev, D]

        ctx_b = self.ctx_proj(ctx).unsqueeze(1)      # [B,1,D]
        aud_b = self.audio_proj(audio).unsqueeze(1)  # [B,1,D]

        E_t0 = base + ctx_b + aud_b                  # [B,Q_ev,D]
        E_t0 = self.out_norm(E_t0)
        return E_t0
```

5. EventDeformableDecoderLayer（概念）

- 各イベントクエリから少数点をサンプリングし、`kv_feats`（= obj_query集合）から疎に情報取得。
- 実装は MSDeformableAttention に置き換える想定。ここでは構造イメージのみ。

```python
class EventDeformableDecoderLayer(nn.Module):
    def __init__(self, dim, num_points):
        super().__init__()
        self.dim = dim
        self.num_points = num_points

        self.offset_mlp = nn.Linear(dim, num_points)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, queries, kv_feats, kv_meta):
        """
        queries:  [B, Q_ev, D]
        kv_feats: [B, L, D]   (L = Q_obj * #window_frames)
        """
        B, Q_ev, D = queries.shape
        _, L, _ = kv_feats.shape

        q_norm = self.norm1(queries)

        # offset -> 重み (簡略版)
        offsets = self.offset_mlp(q_norm)          # [B, Q_ev, num_points]
        weights = F.softmax(offsets, dim=-1)       # [B, Q_ev, num_points]

        # 実用では offsets + kv_meta から座標ベースでサンプル
        # ここでは簡略してランダムインデックスを使用
        idx = torch.randint(0, L, (self.num_points,), device=kv_feats.device)
        sampled = kv_feats[:, idx, :]              # [B, num_points, D]

        # 重み付き集約
        sampled = (weights.unsqueeze(-1) *
                   sampled.unsqueeze(1)).sum(dim=2)  # [B, Q_ev, D]

        updated = queries + self.proj(sampled)

        h = self.norm2(updated)
        h = self.ffn(h)
        updated = updated + h

        return updated
```

6. EventHead

- `[B, T, Q_ev, D_h]` のイベントクエリから、フレームごとに `[B, T, E]` のスコアを出力。
- シンプルに Q_ev 次元で平均（またはattention pool）してからMLP。

```python
class EventHead(nn.Module):
    def __init__(self, dim, num_events, pool_type="mean"):
        super().__init__()
        self.num_events = num_events
        self.pool_type = pool_type

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_events),
        )

    def forward(self, event_queries):
        """
        event_queries: [B, T, Q_ev, D]
        return:
            prob:   [B, T, E]
            logits: [B, T, E]
        """
        B, T, Q_ev, D = event_queries.shape

        if self.pool_type == "mean":
            frame_repr = event_queries.mean(dim=2)  # [B, T, D]
        else:
            frame_repr = event_queries.mean(dim=2)  # 拡張余地あり

        logits = self.mlp(frame_repr)      # [B, T, E]
        prob = torch.sigmoid(logits)       # 多ラベル瞬間イベント想定
        return prob, logits
```

この全体構成のポイント

- シーンモデル:
  - $O_t$（obj_query群）がすでに空間・時間・トラッキングを統合した高次状態。
- イベントモデル:
  - $z_t$（temporal_tokens）と音声$a_t$で「どのタイプのイベントを探すか」をクエリ側に注入。
  - イベントクエリは $E_{t-1}$ を引き継ぎ、時間的連続性を持つ潜在状態。
  - Deformable Attentionは、時間窓 $[t-k, t+k]$ のみの $O_\tau$ を探索対象とし、
    どのオブジェクト・どのタイミングがイベントに対応するかを選択的に取り込む。
- これにより:
  - 低レベル特徴に戻らず、高度に圧縮された表現上で効率的にイベント検知。
  - モデル構造が階層的かつ役割分離され、数学的にも解釈しやすい。

---

#### 数式表現
2. イベントモデル（高次オブジェクト表現上の疎参照）

2.1 入力表現

シーンモデルの出力クエリをオブジェクト表現とみなす:
[
\mathbf{O}*{t}=\mathbf{Q}^{L}*{t}\in\mathbb{R}^{Q\times D},\quad
z_t\in\mathbb{R}^{D}
]
音声潜在 (\mathbf{a}_t\in\mathbb{R}^{D_a}) を時間合わせ（補間）して (D) 次元へ射影:
[
\tilde{\mathbf{a}}_t=\mathbf{W}*a,\text{Align}(\mathbf{a}*{1:T_a})_t\in\mathbb{R}^{D}
]

2.2 再帰イベントクエリ初期化

イベントクエリ数 (Q_{\text{ev}})。学習可能な初期 (\mathbf{E}^{\text{init}}\in\mathbb{R}^{Q_{\text{ev}}\times D})。
前時刻最終 (\mathbf{E}^{L}*{t-1}) と文脈/音声を融合:
[
\begin{aligned}
\mathbf{H}*{\text{prev}} &= \sigma(\mathbf{W}*{\text{ep}}\mathbf{E}^{L}*{t-1}),\quad
\mathbf{H}*{\text{init}} = \sigma(\mathbf{W}*{\text{ei}}\mathbf{E}^{\text{init}}) \
\mathbf{B}^{\text{ev}}*t &= \mathbf{H}*{\text{prev}}\odot \mathbf{E}^{L}*{t-1} + \mathbf{H}*{\text{init}}\odot \mathbf{E}^{\text{init}} \
\mathbf{E}^{0}*t &= \mathrm{LN}\Big(\mathbf{B}^{\text{ev}}*t + \mathbf{1}*{Q*{\text{ev}}} z_t^\top \mathbf{U}*z + \mathbf{1}*{Q_{\text{ev}}} \tilde{\mathbf{a}}_t^\top \mathbf{U}_a\Big)
\end{aligned}
]

2.3 KV はオブジェクト表現のみ（ローカル時間窓）

[
\mathbf{K}^{\text{ev},(t)}=\mathbf{V}^{\text{ev},(t)}=\big[\mathbf{O}*{\tau}\big]*{\tau\in\mathcal{W}(t)}\in\mathbb{R}^{L_{\text{ev}}\times D},\quad L_{\text{ev}}=Q\cdot|\mathcal{W}(t)|
]

2.4 Event-Deformable デコード

シーン側と同様に、各層 (\ell=1,\dots,L_{\text{ev}}):
[
\mathbf{E}^{\ell}_t=\mathrm{FFN}\Big(\mathbf{E}^{\ell-1}_t+\mathbf{W}^{\text{ev}}_o\cdot
\mathrm{DefAtt}\big(\mathbf{E}^{\ell-1}*t,\mathbf{V}^{\text{ev},(t)}\big)\Big)
]
最終 (\mathbf{E}^L_t\in\mathbb{R}^{Q*{\text{ev}}\times D})。

2.5 フレーム別イベント確率

クエリ次元で集約（平均 or アテンションプール）後、多ラベルロジット:
[
\bar{\mathbf{e}}_t=\text{Pool}(\mathbf{E}^L_t)\in\mathbb{R}^{D},\quad
\boldsymbol{\ell}*t=\mathbf{W}*{\text{ev}}\bar{\mathbf{e}}*t+\mathbf{b}*{\text{ev}}\in\mathbb{R}^{E},\quad
\mathbf{p}_t=\sigma(\boldsymbol{\ell}*t)\in[0,1]^E
]
ここで (E) はイベント種類数。出力 (\mathbf{p}*{1:T}\in[0,1]^{T\times E})。

#### データと学習
- ビデオにイベントのタイムスタンプが付与された既存データセットを活用。
- 事前にシーン検知モデルで全データをエンコードして保存し、学習環境から即座に利用可能にする。
- 音声特徴（審判コール、インパクト音、観客リアクション等）はイベント検知モデルの入力として取り込み、映像ベースのシーン検知結果と同期させて利用する。
- 音声埋め込みは `AutoProcessor.from_pretrained("facebook/encodec_24khz")` と `EncodecModel.from_pretrained("facebook/encodec_24khz")` を用いて算出し、時間窓ごとの潜在ベクトルをイベント検知モデルへ供給する。

#### 検知対象イベント
イベント検知モデルは「瞬間」を確率ピークとして出力し、周辺フレームでガウス状に滑らかに遷移する時系列信号を生成する。主に以下の瞬間イベントを検知する。

- **サーブ関連**: トス開始、インパクト（1st/2nd区別）、サーブミス（ネット/アウト）、ダブルフォルト確定、サーブ直後のS+1ショット。
- **リターン関連**: リターン準備モーション（スプリットステップ）、リターンインパクト、リターンミス（ネット/アウト）、R+1ショット。
- **ラリー中ショット**: グラウンドストローク、ボレー、スマッシュ、ドロップ、ロブなど各ショットのインパクト瞬間。
- **ボール接触**: 各バウンド（コート接触）、ネットタッチ、ポール/選手への接触、インプレー終了を示すデッドボール。
- **ポジショニング**: プレーヤーの大きなフットワークイベント（スプリットステップ、ディレクションチェンジ、ネットアプローチ開始）、ネットへの侵入開始、後退開始。
- **クラッチ/スコア更新**: ポイント開始（サーブ前ルーチン完了）、ポイント終了（審判コール）、ブレークポイント/セットポイント/マッチポイント突入、といったスコア状態の変化瞬間。
- **フィジカル兆候**: ジャンプ発生、スライディング、急停止など、疲労やテンポ分析に使う動作瞬間。
- **音声トリガ**: インパクト音のピーク、審判コール、観客のリアクションなど、映像では検知しづらい瞬間を補強する音声イベント。
- **メタタグイベント**: サーバー交代、チェンジエンド開始/終了、サーフェス情報や屋内外タグの付与タイミング（試合ごとの初期イベントとして記録）。

### 3. LLM活用
#### 方針
- RAGを構築して統計データの表現テンプレートを定義し、モデル出力を観察しながら必要に応じてファインチューニングを実施する。

#### モデル選定
- 高度な考察力と要約力を重視し、テニスの戦術的示唆を説明できるモデルを選択する。

## システム全体像
1. シーン検知を動画に適用し、推論結果とエンコーダ特徴を保存する。
2. 保存済みのシーン検知結果と特徴量を入力としてイベント検知モデルを実行し、イベント推論結果を保存する。
3. シーン検知・イベント検知の両結果を統合して統計データを生成し、LLMでユーザー向けレポートに整形する。
