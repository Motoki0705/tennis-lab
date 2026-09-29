# Probabilistic triangulation

CPU float64 の1同期frame用API。`CameraGMM` はsource画素の
`means_px (V,K,2)`、`covariance_px2 (V,K,2,2)`、
`weights (V,K)`、`presence (V)` を受ける。
cameraは既存の `PinholeCamera`（歪み補正後のpinhole座標系）を使用。
出力はworld座標の `GaussianMixture3D` と成分ごとのcamera集合。
単位は呼び出し側のworld座標（ball refinerではm）に従う。
task固有の変換は [BallGMM2D adapter](../../../tasks/ball_refiner/refiner_3d/triangulation.py)
に置き、utilsからtaskへは依存しない。

## 確率モデル

存在確率pᵢに対して、camera集合Sの重みは
`αS = ∏(i∈S) pᵢ ∏(i∉S) (1−pᵢ)`。
各集合の条件付き位置分布を
`qS(x) = p0(x) ∏(i∈S) Gi(πi(x)) / ZS` と正規化してから
`q(x) = ΣS αS qS(x)` を返す。p0は呼び出し側が必ず渡す正定値Gaussian prior。
Sが空ならqS=p0、単眼なら光線方向にprior由来の不確実性が残る。

これは独立Bernoulliの**条件付き融合近似**。
「球の不存在」「検出器の低score」「遮蔽」を同一視しない。
不在cameraの画面外制約とcamera間の存在相関は未モデル化。
異なる集合の2D密度を単位の異なるまま足したり、存在確率を
Mahalanobis誤差の係数に転用したりしない。
`prior_only_probability` は位置情報が無い項の重みであり、球の不存在確率ではない。
#935が検出欠損中にもpresenceの高い分布を生成した場合は、その分布を通常どおり使う。

## 方式A

各Sの全成分組合せに対し、priorと全2D共分散で白色化した
非線形再投影残差を最小化する。共分散は
`(Σ0⁻¹ + Σi Jiᵀ Σi⁻¹ Ji)⁻¹`（Gauss–Newton Laplace近似）。
組合せ重みは2D混合重みの積×Laplace evidence。
evidenceはMAPの残差だけでなく、2D/priorの正規化定数と
`(2π)^(3/2) |Σ3D|^(1/2)` を含む。同じS内で正規化する。

厳密な「Gaussian積」は線形投影の場合だけ。透視投影では非線形最適化と
局所近似であり、残差の二階微分項、単一組合せ内の複数極値、
camera背後/地面下へのGaussian tailの切断は扱わない。
priorの位置・幅は結果に影響するため、設定と実験記録に必ず含める。
背後の最適解・数値失敗・非正定値共分散は例外。成分の黙った破棄やjitter追加はしない。

最悪成分数は `(K+1)^V`（0<pᵢ<1）、全presence=1なら `K^V`。
`LaplaceConfig.max_components` を超える列挙は開始前に拒否する。
Top-K pruning、moment matchingによる単一Gaussian化、point推定への切替は行わない。
`moments()` は成分間分散も含む要約、`sample()` / `log_prob()` は混合全体を扱う。

## 検証と比較

[unit tests](../../../../tests/unit/utils/geometry/test_probabilistic_triangulation.py) は
線形極限の共役Gaussianの平均/共分散・積分evidence、
Monte Carlo較正、成分間の分散、presence周辺化、単眼/全不在、
画素スケールとcamera順の不変性、予算/失敗時のerrorを検証する。
B（適応voxel）とC（2D標本化＋三角測量）は研究用の
[comparison.py](../../../tasks/ball_refiner/refiner_3d/comparison.py) にあり、
本APIの自動fallbackではない。測定・選定理由はknowledgeノードを正本とする。
