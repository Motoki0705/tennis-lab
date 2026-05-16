# CUDA ops

このリポジトリには、PyTorch の標準演算だけで動く reference 実装に加えて、
一部のモデル部品を高速化する optional CUDA extension があります。

現在の CUDA ops は次の 2 種類です。

- `src.utils.models.components.ops.moe._C`: MoE dispatch/combine 用
- `src.utils.models.components.ops.time_local._C`: time-local attention 用

PLCS/BLCS の axial multi-view model では、local time attention を CUDA tensor
上で実行する場合に `time_local` CUDA extension を使います。

## 前提

- CUDA 対応版 PyTorch が `.venv` にインストールされていること
- `nvcc` が使えること
- `CUDA_HOME` が PyTorch から検出できること

確認例:

```bash
.venv/bin/python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("CUDA_HOME:", CUDA_HOME)
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

`nvcc` も確認します。

```bash
nvcc --version
```

## ビルド

CUDA ops は通常の editable install ではビルドされません。
ビルドする場合は `TENNIS_LAB_BUILD_CUDA_OPS=1` を付けて editable install を実行します。

```bash
TENNIS_LAB_BUILD_CUDA_OPS=1 \
  uv pip install -e . --no-build-isolation --python .venv/bin/python
```

ビルドに成功すると、次のような共有ライブラリが生成されます。

```text
src/utils/models/components/ops/moe/_C.so
src/utils/models/components/ops/time_local/_C.so
```

これらは生成物なので git にはコミットしません。

## ロード確認

```bash
.venv/bin/python - <<'PY'
from src.utils.models.components.ops import (
    is_moe_cuda_available,
    is_time_local_cuda_available,
)

print("moe cuda:", is_moe_cuda_available())
print("time-local cuda:", is_time_local_cuda_available())
PY
```

どちらも `True` なら extension を import できています。

## time-local CUDA path の動作確認

`use_cuda=True` を指定すると、extension が未ビルドの場合は fallback せずに失敗します。
これは「本当に CUDA path を使えているか」を確認するために便利です。

```bash
.venv/bin/python - <<'PY'
import torch

from src.utils.models.components.ops.time_local import time_local_attention

query = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)
key = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)
value = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)
valid_mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)

out = time_local_attention(
    query,
    key,
    value,
    valid_mask=valid_mask,
    window_radius=2,
    use_cuda=True,
)
out.square().mean().backward()
torch.cuda.synchronize()

print("out device:", out.device)
print("grad device:", query.grad.device)
PY
```

期待される出力例:

```text
out device: cuda:0
grad device: cuda:0
```

## 実行時の選択

低レベル API の `use_cuda` は次のように振る舞います。

- `use_cuda=True`: CUDA tensor と CUDA extension が必要です。条件を満たさない場合は失敗します。
- `use_cuda=False`: reference 実装を使います。
- `use_cuda=None`: 既定では reference 実装を使います。

環境変数でも制御できます。

- `TENNIS_LAB_USE_TIME_LOCAL_CUDA=1`: `use_cuda=None` の time-local attention で CUDA path を優先します。
- `TENNIS_LAB_FORCE_TIME_LOCAL_REFERENCE=1`: time-local attention で reference 実装を強制します。

PLCS/BLCS の axial multi-view model は、local time attention を CUDA tensor 上で
実行する場合に `use_cuda=True` 相当で呼び出します。そのため、GPU 学習や GPU 推論で
local time attention を使う場合は、事前に `time_local` CUDA extension をビルドしてください。

## トラブルシュート

`Time-local CUDA extension is not available` が出る場合:

- `TENNIS_LAB_BUILD_CUDA_OPS=1` 付きで editable install したか確認する
- `src/utils/models/components/ops/time_local/_C.so` が生成されているか確認する
- `CUDA_HOME` と `nvcc --version` を確認する
- `.venv/bin/python` と `uv pip install --python .venv/bin/python` の対象が同じ環境か確認する

`CUDA_HOME was not found` が出る場合:

- CUDA toolkit が入っているか確認する
- 必要に応じて `CUDA_HOME=/usr/local/cuda` などを設定して再ビルドする

