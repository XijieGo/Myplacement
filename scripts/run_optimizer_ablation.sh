#!/usr/bin/env bash
# Run the retained open-loop baseline and the closed-loop optimizer under exactly
# the same input, resources, seed, and command-line options.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <build-dir> <design.aux> <result-dir> [myplace options...]" >&2
    exit 2
fi

optimizer_build_dir=$1
optimizer_aux_file=$2
optimizer_result_dir=$3
shift 3

optimizer_project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
optimizer_runner="${optimizer_project_dir}/scripts/run_safe_benchmark.sh"
mkdir -p "${optimizer_result_dir}"

"${optimizer_runner}" "${optimizer_build_dir}" "${optimizer_aux_file}" \
    "${optimizer_result_dir}/legacy" "$@" --global-optimizer legacy
"${optimizer_runner}" "${optimizer_build_dir}" "${optimizer_aux_file}" \
    "${optimizer_result_dir}/adaptive" "$@" --global-optimizer adaptive
