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
`qS(x) = p0(x) 1[全i∈Sでdepth_i(x)>0] ∏(i∈S) Gi(πi(x)) / ZS` と正規化してから
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

透視投影では非線形最適化と局所近似になる。Gauss–Newton方向の
fraction-to-boundaryとArmijo line searchを使い、全評価点のdepthを1e-4 world単位より
大きく保つ。prior中心が領域外なら、投影を評価する前に凸な初期化問題を解く。
camera境界1e-3以内、局所depthが3σ未満、評価予算の枯渇、line search失敗は
理由付きの `NonregularComponentError`。境界へ収束した点を通常のLaplace evidenceにしない。
`max_nfev`は残差評価回数の上限で、backtrackingの試行も数える。

厳密なGaussian積になるのは線形極限だけ。残差の二階微分、
1つの成分組合せ内の複数極値は未モデル化であり、大域最適性は保証しない。
正depth領域で積分した場合も、exportするGaussianのtail自体は切断されない。
地面下のtailも残る。priorの位置・幅を必ず設定と記録に含める。
数値失敗や非正定値共分散を黙って修復しない。

## 明示的なA/B併用

`solver.triangulate_hybrid`は`HybridConfig(laplace, volume)`を必須にする別API。
各成分組合せをAで評価し、上記の非正則理由だけをBの適応体積積分へ渡す。
`triangulate_gmm`自体はBへ切り替わらない。成分を削除したりpriorで置き換えたりしない。
Bでも積分できなければframe全体を失敗にする。返却値の`component_methods`に
全成分の方式・理由を保持し、datasetではframe×componentのcodeと集計を保存する。

`volume.py`が比較用Bと併用方式の共通実装。prior中心±指定σの箱を
coarse-to-fineに積分し、未細分化cellの質量も残す。cheiralityはcell中心で判定するため
境界を跨ぐcellには離散化誤差がある。各組合せの積分evidenceと平均・共分散を求め、
その組合せだけを1つのGaussianへ近似する。全組合せ間の多峰性は保持する。
共分散にはuniform cellの幅²/12を含める。箱外prior tailの切断・粗いcell内の
形状誤差・単一組合せ内の非Gaussian形状の喪失は残り、予算増量対照で感度を記録する。
数値的に0になる極小重みも、成分の配列自体は残る。

最悪成分数は `(K+1)^V`（0<pᵢ<1）、全presence=1なら `K^V`。
`LaplaceConfig.max_components` を超える列挙は開始前に拒否する。
Top-K pruning、全混合の単一Gaussian化、point推定への切替は行わない。
`moments()` は成分間分散も含む要約、`sample()` / `log_prob()` は混合全体を扱う。
float32 exportの重み和の丸め誤差は、契約検証後に再正規化する。
成分の選別・閾値処理は含まない。

## 積分の収束判定

`convergence.triangulate_converged`は、明示した複数のvoxel予算でHを再計算する。
正則Laplaceの結果と非正則理由は1frame内でcacheし、Bだけを細分化する。
各隣接段階で全成分のlog evidence絶対差、平均のL2差、共分散の相対Frobenius差
（分母は前後のnormの大きい方）を検査する。重み0へunderflowした成分も検査対象。
さらに、最初/直前/現在の全成分平均と各軸±1周辺標準偏差で混合NLLの最大絶対差を検査する。
GTや選別したseedを停止条件に使わない。単位は呼び出し側のworld単位とnat。

全条件が指定閾値以下なら終了し、未達なら指定capまで進む。最後の分布を必ず返し、
未収束flag、frame/成分別達成差分、各段階の予算と判定を返却する。NLLの良い格子を
選んだり成分を落としたりしない。全B成分を同じ段階で再積分するため、通常のHより高コスト。
数値の正本は呼び出し側の設定であり、ball refinerはdataset_plan.yamlに置く。

これは**有限格子間の安定性チェック**で、連続積分の誤差上界ではない。
有限box外tail、単一productのGaussian moment近似、Laplace誤差は保証しない。
離散化差が小さくても共通の系統誤差は残り得る。cap到達を収束と記録しない。

## 検証と比較

[unit tests](../../../../tests/unit/utils/geometry/test_probabilistic_triangulation.py) は
線形極限の共役Gaussianの平均/共分散・積分evidence、
Monte Carlo較正、成分間の分散、presence周辺化、単眼/全不在、
画素スケールとcamera順の不変性、予算/失敗時のerrorを検証する。
Bは `volume.py`、C（2D標本化＋三角測量）は研究用の
[comparison.py](../../../tasks/ball_refiner/refiner_3d/comparison.py) にある。測定・選定理由はknowledgeノードを正本とする。
