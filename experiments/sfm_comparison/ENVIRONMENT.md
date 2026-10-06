# SfM runtimeの復元記録

2026-10-07に構築したLinux x86-64環境の記録。他環境での動作保証ではない。
Tennis Labの`.venv`は変更せず、NHTとVidMapを別runtimeで実行した。
GPU起動・予算・出力契約は[README](README.md)を参照する。

## 固定sourceとruntime

| checkout | revision |
|---|---|
| NHT | `9493001f1f5650f62e8f9809b023618edcde3f03` |
| [VidMap](https://github.com/cvg/vidmap) | `1a48f2a1c9b59ba1e28bf40eeba1777f5d34ebb1` |
| [COLMAP](https://github.com/colmap/colmap) | `ea063e6874583429143dd5c53d86c3e68c895e8f` |
| [PyCeres](https://github.com/cvg/pyceres) | `86ffa83b7099bdc46a0c8117f606b54f2d83ff03` |
| [PoseLib](https://github.com/PoseLib/PoseLib) | `48b0fdd032534f1d76ffb5bc1536b337d5517c50` |

各checkoutは`.cache/<name>`。VidMapとPoseLibはrecursive submoduleも取得する。
LightGlue、GeoCalibとモデルsourceはVidMap固定版の指定に従った。
NHT baselineはPython 3.11とPyCOLMAP 4.1.1、VidMapはPython 3.11.17とsource版PyCOLMAP 4.3.0。

## VidMapのnative依存

micromamba 2.9.0は`.cache/mamba-bin/bin/micromamba`、rootは`.cache/mamba-root`、
prefixは`.cache/runtimes/vidmap-native`。conda-forgeからPython 3.11、CMake、Ninja、
C++ compiler、Ceres、Eigen、Boost、OpenImageIO、METIS、SuiteSparse、SQLite、GLEW、
glog、gflags、pkg-config、fmt、libgl-devel、libglx-devel、libopengl-develを導入した。
実際の組合せはCeres 2.2.0、Eigen 5.0.1、Boost 1.92.0、conda GCC 15.3。
campaignの`environment/conda-packages-explicit.txt`にpackage URLを保存した。

COLMAPは次の設定で構築した。mapperはCPU、学習済みfrontendはGPUを使う。

```bash
SFM_NATIVE_PREFIX="$PWD/.cache/runtimes/vidmap-native"
.cache/mamba-bin/bin/micromamba run -p "$SFM_NATIVE_PREFIX" cmake \
  -S .cache/colmap -B .cache/colmap-build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$SFM_NATIVE_PREFIX" \
  -DCMAKE_PREFIX_PATH="$SFM_NATIVE_PREFIX" \
  -DCUDA_ENABLED=OFF -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DMVS_ENABLED=OFF -DTESTS_ENABLED=OFF -DONNX_ENABLED=OFF \
  -DCGAL_ENABLED=OFF -DIPO_ENABLED=OFF
.cache/mamba-bin/bin/micromamba run -p "$SFM_NATIVE_PREFIX" \
  cmake --build .cache/colmap-build -j2
.cache/mamba-bin/bin/micromamba run -p "$SFM_NATIVE_PREFIX" \
  cmake --install .cache/colmap-build
```

PyCOLMAPは`.cache/colmap`ルートの`pyproject.toml`から構築した。
`uv pip install --python "$SFM_NATIVE_PREFIX/bin/python" --no-cache .cache/colmap`
に`-Ccmake.define.CMAKE_PREFIX_PATH="$SFM_NATIVE_PREFIX"`、
`-Ccmake.define.pybind11_DIR="$SFM_NATIVE_PREFIX/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"`、
`-Cbuild.tool-args=-j2`を渡した。事前にprefixへpybind11 3.1.0を導入する。
PyCeresも同じCeresに対してwheelを構築した。

PoseLibはGCC 15とEigen 5で警告がerrorとなった。アルゴリズムや警告設定は変えず、
PoseLibだけ`CC=/usr/bin/gcc CXX=/usr/bin/g++`（GCC 13）で再構築した。
`CMAKE_PREFIX_PATH`にprefixとpybind11 package directoryを渡した。
成功したPyCeres 2.7とPoseLib 2.0.5のwheelは`environment/wheels/`に保持する。

VidMapも同じprefix/pybind11を明示し、`uv pip install --no-deps --no-cache -e .cache/vidmap`
で構築した。Python依存は固定版`pyproject.toml`に従い、PyCeres/PoseLibだけは上記wheelを使う。
PyTorch 2.14.0+cu130、TorchVision 0.29.0、xFormers 0.0.35を使用した。
停止時のcache破損によるcuda-bindings/torchvision metadata errorは、当該runtimeだけを
`--no-cache --reinstall`で復元した。最終的に`uv pip check`を通過した。

`environment/vidmap-requirements.txt`は実際のfreezeで、local wheel/editableパスも含む。
そのまま他machineへ渡すlockfileではない。同じsource revisionからnative依存を構築する。
runtimeの変更が生じた場合は別の実験条件として記録する。

## 重み・cache・起動

`prefetch_vidmap.py`はCPUで6個のcheckpoint/configを取得し、固定SHA-256を照合する。
取得前にDドライブの空き112 GiB以上を要求する。合計は約8.3 GiB。
実際のpath/byte数/hashは`environment/checkpoints.json`に保存した。
GPU jobの`HF_HOME`と`TORCH_HOME`は`.cache/vidmap-model-cache`配下へ明示する。
`HF_HUB_DISABLE_XET=1`、Torch Inductor cacheは`.cache/vidmap-compile`、
compile threadsは2、BLAS/OpenMP threadsは4。
これとは別に、VidMapの保存済みgraph/moduleは初回実行で既定の
`~/.cache/vidmap/{romav2,da3}`へ書かれることを実ログで確認した。
後続runはこのcacheも再利用する。配置を変える場合は公式の
`VIDMAP_ROMAV2_CACHE_DIR`と`VIDMAP_DA3_CACHE_DIR`を明示し、cache条件の変更を記録する。

公式`python -m vidmap.run --input_data ... --output ... --device cuda`を使用する。
native Ceres/CHOLMODをTorchより先にloadする公式の順序を保つ。
12枚sanityと90枚比較は別output・queue job・run IDとし、必要artifactは`rec/images.bin`。
既存outputを`--overwrite`で再利用しない。

sanityは`frontend=uncalib/base`、`mapping=uncalib/base`。
NHT seed 42と、VidMap global positioning seed 1・keypoint seed `42 + keyframe_id`は区別する。
小tensorのCUDA probe成功はfull VidMapのVRAM適合・幾何精度を意味しない。

## 共通のCPU評価

同じNHT evaluator sourceとVidMap側のPyCOLMAPで両モデルを読む。入力hash・原寸cameraを確認し、
最終geometryから点ごとの残差を再計算したコピーを評価する。元モデルは変更しない。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .cache/runtimes/vidmap-native/bin/python -m experiments.sfm_comparison.evaluate_model \
  --model /path/to/model --image-dir /path/to/frozen/images \
  --manifest /path/to/manifest.json --evaluator-root .cache/nht \
  --output /path/to/new/evaluation-directory
```

sanityの母数は12、共同比較の母数は90。CLIのgeneric gateとNHTのshort-clip gateは異なるので、
`accepted`の差をアルゴリズム差として扱わない。VidMapのrecは選択keyframeのみなので、
登録率は入力画像全体に対するpose供給範囲として解釈する。再計算した点ごとの内部誤差と、
独立pose/ground GTの誤差は別物である。
