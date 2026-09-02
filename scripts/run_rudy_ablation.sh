#!/usr/bin/env bash
# Reproducible RUDY-energy comparison. Every case uses the same initial
# placement and a fixed held-out proxy grid; only the energy model changes.
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <build-dir> <design.aux> <result-dir> <gpu-device-1..4> [myplace options...]" >&2
    exit 2
fi

rudy_build_dir=$1
rudy_aux_file=$2
rudy_result_dir=$3
rudy_gpu_device=$4
shift 4

rudy_binary="${rudy_build_dir}/myplace"
if [[ ! -x "${rudy_binary}" ]]; then
    echo "Missing executable: ${rudy_binary}" >&2
    exit 2
fi
if [[ ! -f "${rudy_aux_file}" ]]; then
    echo "Missing BookShelf AUX file: ${rudy_aux_file}" >&2
    exit 2
fi
if [[ ! "${rudy_gpu_device}" =~ ^[1-4]$ ]]; then
    echo "GPU device must be in 1..4 on this shared server." >&2
    exit 2
fi

# The CUDA allocation cap is deliberately fixed at 10 GiB.  Refuse to start
# unless the selected shared card can still retain the user's 15 GiB margin.
rudy_limit_gib=${RUDY_GPU_MEMORY_LIMIT_GIB:-10}
if [[ ! "${rudy_limit_gib}" =~ ^[1-9][0-9]*$ ]] || (( rudy_limit_gib > 10 )); then
    echo "RUDY_GPU_MEMORY_LIMIT_GIB must be an integer from 1 through 10." >&2
    exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    rudy_free_mib=$(nvidia-smi --id="${rudy_gpu_device}" --query-gpu=memory.free \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    rudy_required_mib=$(( (rudy_limit_gib + 15) * 1024 ))
    if [[ "${rudy_free_mib}" =~ ^[0-9]+$ ]] && (( rudy_free_mib < rudy_required_mib )); then
        echo "GPU ${rudy_gpu_device} has ${rudy_free_mib} MiB free; need ${rudy_required_mib} MiB to retain 15 GiB." >&2
        exit 2
    fi
fi

mkdir -p "${rudy_result_dir}"

rudy_common=(
    --initial quadratic --iterations "${RUDY_ITERATIONS:-160}" --bins "${RUDY_DENSITY_BINS:-64}" --seed "${RUDY_SEED:-2026}"
    --density-field neumann --rudy-bins "${RUDY_BINS:-96}" --rudy-validation-bins "${RUDY_VALIDATION_BINS:-128}"
    --rudy-capacity-factor "${RUDY_CAPACITY_FACTOR:-1.0}" --rudy-validation-capacity-factor 1.5
    --rudy-start-overflow "${RUDY_START_OVERFLOW:-0.20}" --rudy-ramp-iters "${RUDY_RAMP_ITERS:-24}"
    --compute-backend cuda --gpu-device "${rudy_gpu_device}" --gpu-memory-limit-gib "${rudy_limit_gib}"
    --no-bmp --no-gds
)

run_case() {
    local rudy_case_name=$1
    local rudy_model=$2
    local rudy_weight=$3
    shift 3
    "${rudy_binary}" "${rudy_aux_file}" --output "${rudy_result_dir}/${rudy_case_name}" \
        "${rudy_common[@]}" --routability-model "${rudy_model}" --rudy-weight "${rudy_weight}" "$@"
}

# The monitor baseline calibrates and reports the same proxy but has zero
# force.  It is the fair control for the three objective variants.
run_case baseline_monitor rudy_hinge_l2 0.0 "$@"
run_case hinge_l2 rudy_hinge_l2 "${RUDY_WEIGHT:-0.45}" "$@"
run_case softplus_l2 rudy_softplus_l2 "${RUDY_WEIGHT:-0.45}" "$@"
run_case hinge_l4 rudy_hinge_l4 "${RUDY_WEIGHT:-0.45}" "$@"
