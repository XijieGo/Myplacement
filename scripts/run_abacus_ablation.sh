#!/usr/bin/env bash
# Reproducible greedy-vs-Abacus legalization experiment.
#
# Usage:
#   ./scripts/run_abacus_ablation.sh <design.aux> <global-iters> <output-dir> [extra myplace options...]
#
# The two runs intentionally share every placement option and differ only in
# --legalizer.  The script rejects a comparison if their global histories do
# not match byte-for-byte.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <design.aux> <global-iters> <output-dir> [extra myplace options...]" >&2
    exit 64
fi

aux_path=$1
iterations=$2
output_root=$3
shift 3

binary=${MYPLACEMENT_BIN:-./build/myplace}
if [[ ! -x "$binary" ]]; then
    echo "MyPlacement executable not found: $binary" >&2
    exit 66
fi

common_options=(
    --initial quadratic
    --iterations "$iterations"
    --bins 64
    --density-field neumann
    --seed 2026
    --no-bmp
    --no-gds
    "$@"
)

"$binary" "$aux_path" --output "$output_root/greedy" "${common_options[@]}" --legalizer greedy
"$binary" "$aux_path" --output "$output_root/abacus" "${common_options[@]}" --legalizer abacus

if ! cmp -s "$output_root/greedy/global_history.csv" "$output_root/abacus/global_history.csv"; then
    echo "Refusing to compare: global histories differ." >&2
    exit 2
fi

echo "Global histories are byte-identical; only legalization differs."
awk -F= '
    FNR == NR { greedy[$1] = $2; next }
    { abacus[$1] = $2 }
    END {
        hpwl_delta = 100.0 * (abacus["hpwl"] - greedy["hpwl"]) / greedy["hpwl"]
        movement_delta = 100.0 * (abacus["standard_cell_total_displacement"] - greedy["standard_cell_total_displacement"]) / greedy["standard_cell_total_displacement"]
        printf "HPWL: %.12g -> %.12g (%+.3f%%; negative is better)\n", greedy["hpwl"], abacus["hpwl"], hpwl_delta
        printf "standard-cell total movement: %.12g -> %.12g (%+.3f%%; negative is better)\n", greedy["standard_cell_total_displacement"], abacus["standard_cell_total_displacement"], movement_delta
        printf "legalizer time: %.6fs -> %.6fs\n", greedy["legalization_elapsed_seconds"], abacus["legalization_elapsed_seconds"]
        printf "legal: greedy=%s, abacus=%s\n", greedy["legal"], abacus["legal"]
    }
' "$output_root/greedy/overview.txt" "$output_root/abacus/overview.txt"
