# CLIP-ReID TensorRT計測の再現手順

#2297 のCLIP-ReID ViT-B/16（Market-1501）向け。速度・VRAM・精度・GPU内部指標の定義は [比較プロトコル](PROTOCOL.md) を参照。

## 専用環境

計測環境は Ubuntu 24.04 x86-64 / WSL2、RTX 5060 Ti 16 GiB（CC 12.0）、CUDA Toolkit 13.0、CPython 3.11.15。PyTorch本体は `2.13.0+cu130`、TorchVisionは `0.28.0`。TensorRTはCUDA 13向け `10.14.1.48.post1` Python wheelと、同版C++ `trtexec` のCUDA 13.0 Debianアーカイブを使う。ModelOpt 0.48.0 のONNX PTQで ONNX Runtime のCPU EPを明示する。ONNX・ORT・CUDA wheelを含む全Pythonパッケージの版は [requirements.txt](requirements.txt)。Nsight Compute 2025.3.1はCUDA Toolkitのものを使い、Nsight Systems 2026.5.2は専用環境に展開する。

リポジトリのworktree rootで実行する。インストーラーはrepo外の `~/.venvs/tennis-lab-trt` のみに依存を同期する。 `curl`、`dpkg-deb`、`uv` が必要。

```bash
bundle=knowledge/runs/run-i2297-clipreid-tensorrt
trt_exp_env="$HOME/.venvs/tennis-lab-trt"
bash "$bundle/install-env.sh"
```

[TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)・[PyTorch CUDA 13 wheels](https://pytorch.org/get-started/previous-versions/)

## 入力の展開

`data/tennis_multiview/processed/meiji_4cam/dataset` と `ckpt/player_association/person_vit_clip_reid.pth` の手元資産を使う。場所が異なる場合は `inputs.py --dataset` と `benchmark.py --checkpoint` を指定する。クリーンな出力ディレクトリを条件ごとに使う。

```bash
out=outputs/trt-phase1a-reproduction
PYTHONPATH="$PWD" "$trt_exp_env/bin/python" -B "$bundle/inputs.py" --output "$out/inputs"
```

検出行・box・clip内IDは [固定入力一覧](inputs.json) に含む。元のstoreがある場合は `pin_inputs.py` でも一覧を生成できる。 `--eval-000-store` と `--eval-037-store` は #2238 の `precision/full/fp32/<clip>/store`。 `--calibration-027-store` は #1752 の `video-batch-player-c9-main-20261119/replay/clips/video_000/clip_027`。準備済みの入力配列は [tensor lock](tensor-lock.json) と照合する。

## キューで計測

`TRAINING_QUEUE_DIR` にメインworktreeの共有キューを指定する。各jobは `--resource all` で実行し、条件0の結果を後続4条件から参照する。以下は同じworktreeからの一連の登録例。環境疎通だけなら最初のjobを登録する。Codexから登録する場合は `CODEX_THREAD_ID` を使用し、他のagentはtraining queue skillのprovider/session指定に読み替える。

```bash
queue=.agents/skills/training-queue/scripts/training_queue.sh
export TRAINING_QUEUE_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"
registration=(--resource all --issue 2297)
if [[ -n "${CODEX_THREAD_ID:-}" ]]; then
    registration+=(--provider codex --session "$CODEX_THREAD_ID")
fi

printf -v command '%q ' env "PYTHONPATH=$PWD" "$trt_exp_env/bin/python" -B \
    "$bundle/environment_probe.py" --output "$out/environment"
bash "$queue" add "$command" --name i2297-environment "${registration[@]}"

for condition in 0 1 2 3a 3b; do
    reference=()
    if [[ "$condition" != 0 ]]; then
        reference=(--reference "$out/0")
    fi
    printf -v command '%q ' env "PYTHONPATH=$PWD" "$trt_exp_env/bin/python" -B \
        "$bundle/benchmark.py" --model clipreid_vitb16_market1501 \
        --condition "$condition" --inputs "$out/inputs" --output "$out/$condition" "${reference[@]}"
    bash "$queue" add "$command" --name "i2297-clipreid-$condition" "${registration[@]}"
done
```

キューが停止中であれば共有workerを起動する。 `status.json` と `result.json` で各条件の計測・収集状態を確認する。終了値2は無効なCLI入力。変換や実行の失敗は終了時に非ゼロを返し、 `convert.log` または `measure.log` と `status.json` を残す。終了値3の `profile_incomplete` は速度・メモリ・精度の収集後に指定プロファイラの指標が欠けた状態を示す。個別プロファイラのログと結果はその条件の出力先に残る。 [training queue](../../../.agents/skills/training-queue/SKILL.md)

`ERR_NVGPUCTRPERM` の解消には Windows 側 NVIDIA Control Panel の権限設定が必要。管理者が Desktop メニューで開発者設定を表示し、Developer の Manage GPU Performance Counters で利用者にアクセスを許可する。設定後は同じキュー登録コマンドで `--profiles ncu --profile-from "$out/$condition" --output "$out/ncu-$condition"` を指定する。前回のエンジンと測定結果を照合し、同じ代表入力のカウンタだけを追測する。 [NVIDIAの手順](https://developer.nvidia.com/ERR_NVGPUCTRPERM)

## CPU検証

```bash
PYTHONPATH=".:$bundle" .venv/bin/python -B -m pytest -n0 \
    "$bundle/test_protocol.py" tests/unit/tasks/player_association/test_appearance_sampling.py
```

## 照合できる計測記録

[環境疎通](environment.json)、[5条件の生データと集約](metrics.json)、[検証とエラーの抜粋](validation.json)。 `trtexec --version` はこの版でバナーとビルド番号を表示し、モデル未指定で終了値1を返した。FP8の疎通用planを `--loadEngine=... --skipInference` で読ませると、同じバナーを出して終了値0になった。

INT8の誤差を変換段階ごとに照合する場合、計測済みの `0`、`3a`、`3b` と `inputs` を含む出力先に対して次を実行する。評価の固定順から20件ごとに抽出した12 cropで、ONNX RuntimeのCPU出力と保存済みのPyTorch・TensorRT出力のcosineを表示する。

```bash
"$trt_exp_env/bin/python" -B "$bundle/diagnose_onnx.py" --output-root "$out"
```

後続モデルは [比較プロトコル](PROTOCOL.md) の採取位置に合わせて、入出力・変換・精度比較を接続する。同じ時間計測とNVTX範囲を使うと、`profiles.py` の収集・集約をそのまま使用できる。
