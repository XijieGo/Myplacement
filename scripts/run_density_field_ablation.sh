#!/usr/bin/env bash
# Run a reproducible Neumann-vs-periodic density-field A/B experiment.  Options
# after <result-dir> are forwarded identically to both runs.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <build-dir> <design.aux> <result-dir> [myplace options...]" >&2
    exit 2
fi

ablation_build_dir=$1
ablation_aux_file=$2
ablation_result_dir=$3
shift 3

ablation_project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ablation_runner="${ablation_project_dir}/scripts/run_safe_benchmark.sh"
mkdir -p "${ablation_result_dir}"

"${ablation_runner}" "${ablation_build_dir}" "${ablation_aux_file}" \
    "${ablation_result_dir}/neumann" "$@" --density-field neumann
"${ablation_runner}" "${ablation_build_dir}" "${ablation_aux_file}" \
    "${ablation_result_dir}/periodic" "$@" --density-field periodic
