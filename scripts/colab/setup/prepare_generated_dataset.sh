#!/usr/bin/env bash

# Source-only module for generating a BLCS or PLCS dataset when absent.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[prepare_generated_dataset] This file is a setup module and must be sourced by a train script." >&2
    exit 2
fi

prepare_generated_dataset() {
    local target="$1"
    local repo_root="$2"
    local dataset_dir="$3"
    shift 3

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

    if [[ -f "${dataset_dir}/meta.json" ]]; then
        echo "[prepare_generated_dataset] dataset already exists: ${dataset_dir}"
        return 0
    fi

    echo "[prepare_generated_dataset] generating ${target} dataset: ${dataset_dir}"
    (
        cd "${repo_root}"
        python -m "${module}" "run.output_dir=${dataset_dir}" "$@"
    )
    echo "[prepare_generated_dataset] done."
}
