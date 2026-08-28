#!/usr/bin/env bash

# Source-only module for staging Colab datasets and DINOv3 assets from Drive.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[prepare_archive_dataset] This file is a setup module and must be sourced by a train script." >&2
    exit 2
fi

_prepare_dinov3_submodule() {
    local repo_root="$1"

    echo "[prepare_archive_dataset] preparing DINOv3 submodule..."
    if [[ ! -d "${repo_root}/third_party/dinov3/dinov3" ]]; then
        if ! git -C "${repo_root}" submodule update --init third_party/dinov3; then
            echo "[prepare_archive_dataset] SSH submodule update failed; retrying DINOv3 over HTTPS..."
            git -C "${repo_root}" submodule deinit -f third_party/dinov3 || true
            rm -rf "${repo_root}/.git/modules/third_party/dinov3" "${repo_root}/third_party/dinov3"
            git -C "${repo_root}" config -f .gitmodules submodule.third_party/dinov3.url \
                https://github.com/Motoki0705/dinov3.git
            git -C "${repo_root}" config submodule.third_party/dinov3.url \
                https://github.com/Motoki0705/dinov3.git
            git -C "${repo_root}" submodule sync third_party/dinov3
            git -C "${repo_root}" submodule update --init third_party/dinov3
        fi
    else
        echo "[prepare_archive_dataset] DINOv3 submodule already initialized."
    fi

    if [[ ! -d "${repo_root}/third_party/dinov3/dinov3" ]]; then
        echo "[prepare_archive_dataset] DINOv3 submodule is missing after initialization." >&2
        return 1
    fi
}

_stage_dinov3_checkpoint() {
    local src="$1"
    local dest_dir="$2"
    local dest="${dest_dir}/$(basename "${src}")"

    if [[ ! -f "${src}" ]]; then
        echo "[prepare_archive_dataset] missing DINOv3 checkpoint: ${src}" >&2
        return 1
    fi

    mkdir -p "${dest_dir}"
    if [[ ! -f "${dest}" || "${src}" -nt "${dest}" ]]; then
        echo "[prepare_archive_dataset] copying DINOv3 checkpoint $(basename "${src}")..."
        cp -av "${src}" "${dest}"
    else
        echo "[prepare_archive_dataset] DINOv3 checkpoint already exists: ${dest}"
    fi
}

prepare_archive_dataset() {
    local target="$1"
    local repo_root="$2"
    local drive_data="${DRIVE_DATA:-/content/drive/MyDrive/tennis_lab/data}"
    local cache_dir="${CACHE_DIR:-/content/drive_upload_archives}"
    local data_dir="${DATA_DIR:-${repo_root}/data}"
    local dinov3_ckpt_name="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    local dinov3_ckpt="${DINOV3_CKPT:-${drive_data}/${dinov3_ckpt_name}}"
    local dinov3_ssl_ckpt_name="dinov3_vitb16_tennis_ssl_merged.pth"
    local dinov3_ssl_ckpt="${DINOV3_SSL_CKPT:-${drive_data}/${dinov3_ssl_ckpt_name}}"
    local dinov3_dest_dir="${repo_root}/third_party/dinov3/checkpoints"
    local archives=()

    case "${target}" in
        ball)
            archives=("tennis.tar.zst")
            ;;
        court)
            archives=("court.tar.zst")
            ;;
        synthetic_court_v2)
            archives=("synthetic_court_v2.tar.zst")
            ;;
        court_query_issue790_v3)
            archives=("court_query_issue790_v3.tar.zst")
            ;;
        plcs)
            archives=("smplx.tar.zst" "smplh.tar.zst" "ACCAD.tar.zst")
            ;;
        dinov3_ssl)
            archives=("dinov3_ssl.tar.zst")
            ;;
        *)
            echo "[prepare_archive_dataset] unknown target: ${target}" >&2
            return 2
            ;;
    esac

    if [[ ! -d /content/drive/MyDrive ]]; then
        echo "[prepare_archive_dataset] Google Drive is not mounted at /content/drive/MyDrive." >&2
        return 1
    fi
    if ! command -v zstd >/dev/null 2>&1; then
        echo "[prepare_archive_dataset] zstd is required; install_colab_dependencies must run first." >&2
        return 1
    fi

    echo "[prepare_archive_dataset] target: ${target}"
    echo "[prepare_archive_dataset] drive data: ${drive_data}"
    echo "[prepare_archive_dataset] cache dir: ${cache_dir}"
    echo "[prepare_archive_dataset] data dir: ${data_dir}"
    mkdir -p "${cache_dir}" "${data_dir}"

    local archive src dst
    for archive in "${archives[@]}"; do
        src="${drive_data}/${archive}"
        dst="${cache_dir}/${archive}"
        if [[ ! -f "${src}" ]]; then
            echo "[prepare_archive_dataset] missing source archive: ${src}" >&2
            return 1
        fi
        if [[ ! -f "${dst}" || "${src}" -nt "${dst}" ]]; then
            echo "[prepare_archive_dataset] copying ${archive}..."
            cp -av "${src}" "${dst}"
        else
            echo "[prepare_archive_dataset] cache already exists: ${dst}"
        fi
    done

    for archive in "${archives[@]}"; do
        echo "[prepare_archive_dataset] extracting ${archive} to ${data_dir}..."
        tar -I zstd -xf "${cache_dir}/${archive}" -C "${data_dir}"
    done

    if [[ "${target}" == "ball" || "${target}" == "court" \
          || "${target}" == "synthetic_court_v2" \
          || "${target}" == "court_query_issue790_v3" \
          || "${target}" == "dinov3_ssl" ]]; then
        _prepare_dinov3_submodule "${repo_root}"
        _stage_dinov3_checkpoint "${dinov3_ckpt}" "${dinov3_dest_dir}"
    fi
    if [[ "${target}" == "ball" || "${target}" == "court" ]]; then
        _stage_dinov3_checkpoint "${dinov3_ssl_ckpt}" "${dinov3_dest_dir}"
    fi

    echo "[prepare_archive_dataset] done."
}
