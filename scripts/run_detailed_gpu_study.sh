#!/usr/bin/env bash
# Run the A100/A800 four-cell detailed-placement experiment only when a shared
# GPU is genuinely idle.  This is intentionally stricter than a memory-only
# preflight: abundant free VRAM does not make a 100%-busy GPU available.
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <build-dir> <design.aux> <output-dir> <gpu-device-1..4> [myplace options...]" >&2
    exit 2
fi

detail_build_dir=$1
detail_aux_file=$2
detail_output_dir=$3
detail_gpu_device=$4
shift 4

detail_binary="${detail_build_dir}/myplace"
if [[ ! -x "${detail_binary}" ]]; then
    echo "Missing executable: ${detail_binary}" >&2
    exit 2
fi
if [[ ! -f "${detail_aux_file}" ]]; then
    echo "Missing BookShelf AUX file: ${detail_aux_file}" >&2
    exit 2
fi
if [[ ! "${detail_gpu_device}" =~ ^[1-4]$ ]]; then
    echo "GPU device must be in 1..4 on this shared server." >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for the shared-GPU idle preflight." >&2
    exit 2
fi

# The detailed kernel itself needs far less than this on the course cases, but
# keeping a configurable explicit cap makes the run auditable and lets the
# existing global CUDA backend share the same envelope.
detail_memory_limit_gib=${DETAIL_GPU_MEMORY_LIMIT_GIB:-2}
detail_idle_utilization=${DETAIL_GPU_MAX_UTILIZATION_PERCENT:-5}
if [[ ! "${detail_memory_limit_gib}" =~ ^[1-9][0-9]*$ ]] || (( detail_memory_limit_gib > 10 )); then
    echo "DETAIL_GPU_MEMORY_LIMIT_GIB must be an integer from 1 through 10." >&2
    exit 2
fi
if [[ ! "${detail_idle_utilization}" =~ ^[0-9]+$ ]] || (( detail_idle_utilization > 25 )); then
    echo "DETAIL_GPU_MAX_UTILIZATION_PERCENT must be an integer from 0 through 25." >&2
    exit 2
fi

# In addition to the placement process's own 4 GiB in-process reserve, retain
# 15 GiB for the shared machine.  Two samples avoid launching during a brief
# utilization dip between kernels.
detail_required_free_mib=$(( (detail_memory_limit_gib + 15) * 1024 ))
for detail_sample in 1 2; do
    detail_state=$(nvidia-smi --id="${detail_gpu_device}" \
        --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)
    IFS=, read -r detail_free_mib detail_utilization <<< "${detail_state}"
    detail_free_mib=${detail_free_mib//[[:space:]]/}
    detail_utilization=${detail_utilization//[[:space:]]/}
    if [[ ! "${detail_free_mib}" =~ ^[0-9]+$ || ! "${detail_utilization}" =~ ^[0-9]+$ ]]; then
        echo "Could not parse GPU ${detail_gpu_device} state: ${detail_state}" >&2
        exit 2
    fi
    if (( detail_free_mib < detail_required_free_mib )); then
        echo "GPU ${detail_gpu_device} has ${detail_free_mib} MiB free; need ${detail_required_free_mib} MiB." >&2
        exit 2
    fi
    if (( detail_utilization > detail_idle_utilization )); then
        echo "GPU ${detail_gpu_device} is ${detail_utilization}% busy; idle threshold is ${detail_idle_utilization}%." >&2
        exit 2
    fi
    if (( detail_sample == 1 )); then sleep 2; fi
done

detail_iterations=${DETAIL_ITERATIONS:-160}
detail_bins=${DETAIL_DENSITY_BINS:-64}
detail_seed=${DETAIL_SEED:-2026}
detail_passes=${DETAIL_PASSES:-2}

exec "$(dirname "$0")/run_safe_benchmark.sh" "${detail_build_dir}" "${detail_aux_file}" "${detail_output_dir}" \
    --initial quadratic --iterations "${detail_iterations}" --bins "${detail_bins}" \
    --density-field neumann --seed "${detail_seed}" \
    --compute-backend cuda --gpu-device "${detail_gpu_device}" \
    --gpu-memory-limit-gib "${detail_memory_limit_gib}" \
    --detailed-placement window --detailed-backend cuda --detailed-passes "${detail_passes}" \
    --detailed-window 4 --no-bmp --no-gds "$@"
