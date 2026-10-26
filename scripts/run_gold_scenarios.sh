#!/usr/bin/env bash
# Run Python gold scenarios S00–S07 locally (optional GNU parallel JOBS).
# Usage:
#   ./scripts/run_gold_scenarios.sh              # full week
#   ./scripts/run_gold_scenarios.sh SMOKE        # npoints=100
#   ./scripts/run_gold_scenarios.sh S00 S01
#   JOBS=2 BD_NPOINTS=100 ./scripts/run_gold_scenarios.sh
#
# Env: BD_TF, BD_DT, BD_NPOINTS, BD_SEED, BD_OUTDIR, JOBS, BD_LOG_DIR

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

JOBS="${JOBS:-4}"
BD_NPOINTS="${BD_NPOINTS:-}"
BD_TF="${BD_TF:-}"
BD_DT="${BD_DT:-30}"
BD_SEED="${BD_SEED:-42}"
BD_OUTDIR="${BD_OUTDIR:-data/results_py}"

if [[ $# -eq 1 && "$1" == "SMOKE" ]]; then
  SCENARIOS=(S00 S01 S02 S03 S04 S05 S06 S07)
  BD_NPOINTS="${BD_NPOINTS:-100}"
  echo "SMOKE mode: BD_NPOINTS=${BD_NPOINTS}"
elif [[ $# -gt 0 ]]; then
  SCENARIOS=("$@")
else
  SCENARIOS=(S00 S01 S02 S03 S04 S05 S06 S07)
fi

mkdir -p "${BD_OUTDIR}" logs

if [[ -n "${BD_LOG_DIR:-}" ]]; then
  LOG_DIR="${BD_LOG_DIR}"
else
  LOG_DIR="${ROOT}/logs/gold_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_DIR}"
if [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${ROOT}/${LOG_DIR}"
fi

# Prefer repo .venv so GNU parallel SSH jobs (minimal PATH, often no `uv`)
# still hit scipy/numpy. Fall back to uv / system python3 only if needed.
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY=("${ROOT}/.venv/bin/python" -m bdsim.scenarios)
elif command -v uv >/dev/null 2>&1; then
  PY=(uv run python -m bdsim.scenarios)
else
  echo "error: no ${ROOT}/.venv/bin/python and uv not on PATH" >&2
  echo "  run: uv sync   (or python3 -m venv .venv && pip install -e .)" >&2
  exit 1
fi

extra_args=()
[[ -n "${BD_NPOINTS}" ]] && extra_args+=(--npoints "${BD_NPOINTS}")
[[ -n "${BD_TF}" ]] && extra_args+=(--tf "${BD_TF}")

run_one() {
  local id="$1"
  local log="${LOG_DIR}/${id}.log"
  {
    echo "=== ${id} $(date -Iseconds) host=$(hostname -s 2>/dev/null || hostname) ==="
    "${PY[@]}" "${id}" --outdir "${BD_OUTDIR}" --seed "${BD_SEED}" --dt "${BD_DT}" \
      "${extra_args[@]}"
    echo "OK ${id}"
  } 2>&1 | tee "${log}"
}

{
  echo "=== bdsim gold suite $(date -Iseconds) ==="
  echo "SCENARIOS=${SCENARIOS[*]}"
  echo "OUTDIR=${BD_OUTDIR}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "JOBS=${JOBS}"
} | tee -a "${LOG_DIR}/suite.log"

if command -v parallel >/dev/null 2>&1 && [[ "${JOBS}" -gt 1 && ${#SCENARIOS[@]} -gt 1 ]]; then
  # Export for parallel local jobs
  export LOG_DIR BD_OUTDIR BD_SEED BD_DT
  export BD_NPOINTS BD_TF
  # shellcheck disable=SC2034
  export PY0="${PY[0]}"
  export PY1="${PY[1]:-}"
  export PY2="${PY[2]:-}"
  export PY3="${PY[3]:-}"
  export EXTRA_ARGS_STR="${extra_args[*]-}"
  printf '%s\n' "${SCENARIOS[@]}" | parallel -j "${JOBS}" --line-buffer \
    'id={};
     log="'"${LOG_DIR}"'/${id}.log";
     {
       echo "=== ${id} $(date -Iseconds) ===";
       args=();
       [[ -n "'"${BD_NPOINTS}"'" ]] && args+=(--npoints "'"${BD_NPOINTS}"'");
       [[ -n "'"${BD_TF}"'" ]] && args+=(--tf "'"${BD_TF}"'");
       '"${PY[*]}"' "${id}" --outdir "'"${BD_OUTDIR}"'" --seed "'"${BD_SEED}"'" --dt "'"${BD_DT}"'" "${args[@]}";
       echo "OK ${id}";
     } 2>&1 | tee "${log}"'
else
  for id in "${SCENARIOS[@]}"; do
    run_one "${id}"
  done
fi

echo "Done. Results under ${BD_OUTDIR}/  Logs under ${LOG_DIR}"
