#!/usr/bin/env bash

# Source-only validation helpers for Colab role-root and derived-path settings.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[path_contract] This file is a setup module and must be sourced by a train script." >&2
    exit 2
fi

validate_colab_role_root() {
    local name="$1"
    local value="$2"

    if [[ -z "${value}" || "${value}" =~ ^[[:space:]] || "${value}" =~ [[:space:]]$ ]]; then
        echo "[path_contract] ${name} must be a non-empty trimmed absolute path." >&2
        return 2
    fi
    if [[ "${value}" != /* || "${value}" == / ]]; then
        echo "[path_contract] ${name} must be an absolute path other than '/': ${value}" >&2
        return 2
    fi
}

validate_colab_role_child() {
    local name="$1"
    local value="$2"

    if [[ -z "${value}" || "${value}" =~ ^[[:space:]] || "${value}" =~ [[:space:]]$ ]]; then
        echo "[path_contract] ${name} must be a non-empty trimmed role-relative path." >&2
        return 2
    fi
    if [[ "${value}" == /* ]]; then
        echo "[path_contract] ${name} must be role-relative, not absolute: ${value}" >&2
        return 2
    fi
    case "/${value}/" in
        */./* | */../*)
            echo "[path_contract] ${name} must not contain '.' or '..' components: ${value}" >&2
            return 2
            ;;
    esac
}
