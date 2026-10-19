#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help || "${1:-}" == help ]]; then
    cat <<'EOF'
Usage: scripts/configure.sh [PRESET] [CMAKE_OPTION ...]

Presets:
  cpu-diar
  vulkan-diar
  cuda-diar
  metal-diar

Examples:
  scripts/configure.sh cpu-diar
  scripts/configure.sh cuda-diar -DCMAKE_CUDA_ARCHITECTURES=89

The llama.cpp submodule provides ggml; CMake applies patches/ when enabled by
the selected preset.
EOF
    exit 0
fi

PRESET="${1:-cpu-diar}"
if [[ "$PRESET" == -* ]]; then
    PRESET=cpu-diar
else
    shift || true
fi

case "$PRESET" in
    cpu-diar|vulkan-diar|cuda-diar|metal-diar) ;;
    *)
        echo "error: unknown preset '$PRESET'" >&2
        echo "       run: scripts/configure.sh --help" >&2
        exit 2
        ;;
esac

cd "$ROOT"
if [[ ! -f llama.cpp/ggml/CMakeLists.txt ]]; then
    echo "error: llama.cpp submodule is not initialized; it provides ggml" >&2
    echo "       run: git submodule update --init llama.cpp" >&2
    exit 1
fi

cmake --preset "$PRESET" "$@"
