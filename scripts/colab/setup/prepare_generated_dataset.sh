#!/usr/bin/env bash

# Source-only module for generating a BLCS or PLCS dataset when absent.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[prepare_generated_dataset] This file is a setup module and must be sourced by a train script." >&2
    exit 2
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/path_contract.sh"

prepare_generated_dataset() {
    if [[ "$#" -lt 4 ]]; then
        echo "[prepare_generated_dataset] expected: target repo_root data_root dataset_dir [overrides...]" >&2
        return 2
    fi

    local target="$1"
    local repo_root="$2"
    local data_root="$3"
    local dataset_dir="$4"
    local dataset_path
    shift 4

    validate_colab_role_root DATA_ROOT "${data_root}" || return $?
    validate_colab_role_child DATASET_DIR "${dataset_dir}" || return $?
    dataset_path="${data_root%/}/${dataset_dir}"

    local module
    case "${target}" in
        blcs)
            module="src.tasks.blcs.scripts.generate_dataset"
            ;;
        plcs)
            module="src.tasks.plcs.scripts.generate_dataset"
            ;;
        *)
            echo "[prepare_generated_dataset] unknown target: ${target}" >&2
            return 2
            ;;
    esac

    if [[ -f "${dataset_path}/meta.json" ]]; then
        echo "[prepare_generated_dataset] dataset already exists: ${dataset_path}"
        return 0
    fi

    echo "[prepare_generated_dataset] generating ${target} dataset: ${dataset_path}"
    (
        cd "${repo_root}"
        python -m "${module}" \
            "paths.data_root=${data_root}" \
            "run.output_dir=${dataset_dir}" \
            "$@"
    ) || return $?
    echo "[prepare_generated_dataset] done."
}
