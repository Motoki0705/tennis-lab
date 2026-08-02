#!/usr/bin/env bash

# Source-only module for installing the common Colab dependencies.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[install_deps] This file is a setup module and must be sourced by a train script." >&2
    exit 2
fi

install_colab_dependencies() {
    local repo_root="$1"

    echo "[install_deps] repo root: ${repo_root}"
    if [[ ! -d /content/drive/MyDrive ]]; then
        echo "[install_deps] Google Drive is not mounted at /content/drive/MyDrive." >&2
        echo "[install_deps] Mount Drive before running the train script." >&2
        return 1
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "[install_deps] apt-get is required for the Colab runtime setup." >&2
        return 1
    fi

    echo "[install_deps] installing system dependencies..."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y zstd

    echo "[install_deps] installing Python dependencies..."
    python -m pip install --upgrade pip
    python -m pip install hydra-core pytorch-lightning smplx

    echo "[install_deps] done."
}
