#!/usr/bin/env bash
# Run one placement job inside a conservative process envelope.  The placer itself
# uses serial FFTW plans; taskset also prevents any linked numerical library from
# consuming more than sixteen logical CPUs.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <build-dir> <design.aux> <output-dir> [myplace options...]" >&2
    exit 2
fi

benchmark_build_dir=$1
benchmark_aux_file=$2
benchmark_output_dir=$3
shift 3

benchmark_binary="${benchmark_build_dir}/myplace"
if [[ ! -x "${benchmark_binary}" ]]; then
    echo "Missing executable: ${benchmark_binary}" >&2
    exit 2
fi
if [[ ! -f "${benchmark_aux_file}" ]]; then
    echo "Missing BookShelf AUX file: ${benchmark_aux_file}" >&2
    exit 2
fi

# RLIMIT_AS is deliberately much smaller than the host memory.  It leaves a wide
# safety margin while comfortably covering the largest bundled adaptec1 run.
exec /usr/bin/prlimit --as=17179869184 -- \
    env OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    taskset -c 0-15 "${benchmark_binary}" "${benchmark_aux_file}" \
        --output "${benchmark_output_dir}" "$@"
