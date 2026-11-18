# Meiji人物推論の再現手順

固定8画像・全尺ともに、入力pinにDINOサブモジュールのcommitと
`MultiScaleDeformableAttention.so` のSHA-256がある。再現スクリプトは、固定したsourceから
CPUで拡張をビルドし、pinと一致したバイナリだけを選ぶ。
[smokeのpin](smoke-inputs.json)・[全尺のpin](full-inputs.json)・[元のビルド記録](https://github.com/Motoki0705/tennis-lab/blob/d392dd375776de72f604aca7be0bbb81025b7b2c/knowledge/runs/run-i1752-association-stop-20261118/native-build-info.json)

## DINO拡張の準備

| 項目 | SHA-256が一致したCPUビルドの条件 |
|---|---|
| tennis-lab | smokeの記録commitの `scripts/build_dino_extension.sh` と `setup.py` |
| DINO | 固定入力pinのgitlink。必要に応じて同じcheckoutで `git submodule update --init --checkout -- third_party/DINO` |
| 環境 | Ubuntu 24.04 x86-64。Python 3.11.15のuv配布、setuptools 81.0.0 |
| PyTorch | `torch.__version__` が `2.13.0+cu130`、`torch.version.cuda` が `13.0` |
| CUDA Toolkit | `/usr/local/cuda-13.0` のnvcc 13.0.88（compiler build 36424714_0）、`TORCH_CUDA_ARCH_LIST=12.0`、`MAX_JOBS=2` |
| ホストコンパイラとbuild tools | GCC・G++ 13.3.0-6ubuntu2~24.04.1、GNU ld 2.42（binutils 2.42-4ubuntu2.8）、Ninja 1.11.1 |

既存環境の版を確認する。新しい専用checkoutで `.venv` を用意する場合は、記録commitのlockから
依存を揃え、計測で使ったPyTorchの版をその環境に入れる。
[torchの配布](https://pypi.org/project/torch/2.13.0/)・[torchvisionの配布](https://pypi.org/project/torchvision/0.28.0/)・[setuptoolsの配布](https://pypi.org/project/setuptools/81.0.0/)

```bash
uv sync --locked --python 3.11.15 --no-install-project
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple \
    'torch==2.13.0' 'torchvision==0.28.0' 'setuptools==81.0.0'
```

ELFにはコンパイル時の絶対パスと、NVCCがPIDから付けるホスト側一時ファイル名が入る。
`rebuild-dino.sh` は一時ディレクトリに固定commitのsourceを展開し、Linuxの
user・mount・PID名前空間内だけで記録時の名前を再現する。名前空間の外では、実行元の
checkoutとその `.venv`、指定した出力先を使う。CUDAを隠してビルドとCPU dispatch検査を行い、
SHA-256が一致した場合だけ指定先の `lib/` に置く。既存ファイルを使う場合もSHA-256を照合する。
[CPUビルドと照合](rebuild-dino.sh)

再現bundleの絶対パスを `bundle` に入れ、計測commitのcheckout rootから実行する。
両方の計測に使う出力先を `precision_output` に指定する。

```bash
bash "$bundle/rebuild-dino.sh" "$bundle/smoke-inputs.json" "$precision_output/native-dino"
bash "$bundle/rebuild-dino.sh" "$bundle/full-inputs.json" "$precision_output/native-dino"
```

## 固定入力での再計測

各 `repro.sh` が指定するcommitの専用checkoutを使う。再現用スクリプトはこの記録を含む
checkoutから呼び出し、計測コードと入力pinは記録commitのcheckoutから読む。動画・重み・コート・
driftは入力pinを参照する。キューの
コマンドに共通の絶対出力先 `TENNIS_PRECISION_OUTPUT` を渡し、全尺には固定済みコートと
driftを含む `TENNIS_PRECISION_SHARED_ROOT` も渡す。`TENNIS_PRECISION_DATASET` の既定は
checkout内の `data/tennis_multiview/processed/meiji_4cam/dataset`。

1. 固定8画像の `repro.sh` を記録commitのcheckoutから共有training queueへ `--resource all` で登録し、完了を確認する。
2. 全尺の `repro.sh` を対応するcommitのcheckoutから同じqueueへ登録する。新規計測は引数を省略する。中断前の完了ケースと環境記録を引き継ぐ場合は、同じ出力先で `--resume` を指定する。

[固定8画像](../run-person-stage-precision-smoke-20261118/repro.sh)・[全尺](repro.sh)・[共有キューの手順](../../../.agents/skills/training-queue/SKILL.md)

この手順の検証ではCPUで拡張を新規ビルドし、smokeと全尺の両pinのSHA-256が一致した。記録済みのGPU計測は再実行していない。
