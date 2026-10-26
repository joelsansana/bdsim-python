#!/usr/bin/env bash
# Spread Python gold scenarios across cluster nodes via GNU parallel --sshlogin.
# Topology: config/cluster.yaml (or BD_CLUSTER_CONFIG=path).
#
# Usage:
#   ./scripts/run_gold_scenarios_cluster.sh              # S00–S07 full week
#   ./scripts/run_gold_scenarios_cluster.sh SMOKE        # npoints=100
#   ./scripts/run_gold_scenarios_cluster.sh S00 S01
#   NODES='2/:,2/eugeniu@192.168.1.86' ./scripts/run_gold_scenarios_cluster.sh
#
# Env: BD_TF, BD_DT, BD_NPOINTS, BD_SEED, BD_OUTDIR, NODES, BD_LOG_SYNC_SEC,
#      BD_CLUSTER_CONFIG
# Syncs logs/ and results_py/ from SSH remotes.
# Docs: docs/30-engine/Gold-parity-cluster.md

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results_py logs

BD_NPOINTS="${BD_NPOINTS:-}"
BD_TF="${BD_TF:-}"
BD_DT="${BD_DT:-30}"
BD_SEED="${BD_SEED:-42}"
BD_OUTDIR="${BD_OUTDIR:-results_py}"

CONFIG="${BD_CLUSTER_CONFIG:-$ROOT/config/cluster.yaml}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "error: cluster config not found: ${CONFIG}" >&2
  exit 1
fi
# shellcheck disable=SC1090
eval "$(python3 "$ROOT/config/load_cluster_env.py" "${CONFIG}")"
NODES="${NODES:-$CLUSTER_NODES}"
BD_LOG_SYNC_SEC="${BD_LOG_SYNC_SEC:-$CLUSTER_LOG_SYNC_SEC}"

if [[ $# -eq 1 && "$1" == "SMOKE" ]]; then
  SCENARIOS=(S00 S01 S02 S03 S04 S05 S06 S07)
  BD_NPOINTS="${BD_NPOINTS:-100}"
  echo "SMOKE mode: BD_NPOINTS=${BD_NPOINTS}"
elif [[ $# -gt 0 ]]; then
  SCENARIOS=("$@")
else
  SCENARIOS=(S00 S01 S02 S03 S04 S05 S06 S07)
fi

if [[ -n "${BD_LOG_DIR:-}" ]]; then
  LOG_DIR="${BD_LOG_DIR}"
else
  LOG_DIR="${ROOT}/logs/gold_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_DIR}"
if [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${ROOT}/${LOG_DIR}"
fi

if ! command -v parallel >/dev/null 2>&1; then
  echo "error: GNU parallel not found on PATH" >&2
  exit 1
fi

REMOTES=()
IFS=',' read -ra _NODE_PARTS <<< "${NODES}"
for _part in "${_NODE_PARTS[@]}"; do
  _login="${_part#*/}"
  if [[ -n "${_login}" && "${_login}" != ":" ]]; then
    REMOTES+=("${_login}")
  fi
done

sync_remote() {
  local reason="${1:-periodic}"
  if [[ ${#REMOTES[@]} -eq 0 ]]; then
    return 0
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    return 0
  fi
  local r
  for r in "${REMOTES[@]}"; do
    if rsync -az --timeout=20 "${r}:${LOG_DIR}/" "${LOG_DIR}/" 2>>"${LOG_DIR}/sync.log"; then
      echo "[$(date -Iseconds)] log sync OK from ${r} (${reason})" >>"${LOG_DIR}/sync.log"
    else
      echo "[$(date -Iseconds)] log sync WARN from ${r} (${reason})" >>"${LOG_DIR}/sync.log"
    fi
    if rsync -az --timeout=60 \
      "${r}:${ROOT}/${BD_OUTDIR}/" "${ROOT}/${BD_OUTDIR}/" 2>>"${LOG_DIR}/sync.log"; then
      echo "[$(date -Iseconds)] results sync OK from ${r} (${reason})" >>"${LOG_DIR}/sync.log"
    else
      echo "[$(date -Iseconds)] results sync WARN from ${r} (${reason})" >>"${LOG_DIR}/sync.log"
    fi
  done
}

SYNC_PID=""
cleanup_sync() {
  if [[ -n "${SYNC_PID}" ]] && kill -0 "${SYNC_PID}" 2>/dev/null; then
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
  fi
  sync_remote "final"
}
trap cleanup_sync EXIT

{
  echo "=== bdsim-python gold cluster $(date -Iseconds) ==="
  echo "Config: ${CONFIG}"
  echo "Cluster nodes: ${NODES}"
  echo "Scenarios: ${SCENARIOS[*]}"
  echo "Workdir: ${ROOT}"
  echo "LOG_DIR: ${LOG_DIR}"
  echo "OUTDIR: ${BD_OUTDIR}"
  echo "BD_NPOINTS=${BD_NPOINTS:-<none>}"
  echo "Remotes: ${REMOTES[*]:-<none>}"
} | tee -a "${LOG_DIR}/suite.log"

# Fail fast if a remote lacks the runner or .venv (SSH PATH often has no `uv`).
_preflight_fail=0
for _r in "${REMOTES[@]+"${REMOTES[@]}"}"; do
  _probe="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${_r}" \
    "test -d '${ROOT}' && echo DIR_OK || echo DIR_MISSING; test -x '${ROOT}/scripts/run_gold_scenarios.sh' && echo SCRIPT_OK || echo SCRIPT_MISSING; test -x '${ROOT}/.venv/bin/python' && echo VENV_OK || echo VENV_MISSING; '${ROOT}/.venv/bin/python' -c 'import scipy' 2>/dev/null && echo SCIPY_OK || echo SCIPY_MISSING" \
    2>&1 || echo "SSH_FAIL")"
  if echo "${_probe}" | grep -q SSH_FAIL; then
    echo "error: SSH failed to remote ${_r}" >&2
    _preflight_fail=1
  elif echo "${_probe}" | grep -q DIR_MISSING; then
    echo "error: remote ${_r} missing workdir ${ROOT}" >&2
    _preflight_fail=1
  elif echo "${_probe}" | grep -q SCRIPT_MISSING; then
    echo "error: remote ${_r} missing ${ROOT}/scripts/run_gold_scenarios.sh" >&2
    echo "  forge checkout looks stale -- sync/pull bdsim-python on that host (or rsync scripts/)." >&2
    _preflight_fail=1
  elif echo "${_probe}" | grep -q VENV_MISSING; then
    echo "error: remote ${_r} missing ${ROOT}/.venv/bin/python" >&2
    echo "  on ${_r}: cd ${ROOT} && uv sync" >&2
    _preflight_fail=1
  elif echo "${_probe}" | grep -q SCIPY_MISSING; then
    echo "error: remote ${_r} .venv cannot import scipy" >&2
    echo "  on ${_r}: cd ${ROOT} && uv sync" >&2
    _preflight_fail=1
  fi
done
if [[ "${_preflight_fail}" -ne 0 ]]; then
  echo "error: cluster preflight failed" >&2
  exit 2
fi

if [[ ${#REMOTES[@]} -gt 0 && "${BD_LOG_SYNC_SEC}" != "0" ]]; then
  (
    while true; do
      sleep "${BD_LOG_SYNC_SEC}"
      sync_remote "periodic"
    done
  ) &
  SYNC_PID=$!
fi

# Each job runs a single scenario with JOBS=1 on the worker.
set +e
printf '%s\n' "${SCENARIOS[@]}" | parallel \
  --sshlogin "$NODES" \
  --workdir "$ROOT" \
  --joblog "${LOG_DIR}/cluster_joblog.tsv" \
  --tag \
  --line-buffer \
  "JOBS=1 BD_TF='${BD_TF}' BD_DT='${BD_DT}' BD_NPOINTS='${BD_NPOINTS}' BD_SEED='${BD_SEED}' BD_OUTDIR='${BD_OUTDIR}' BD_LOG_DIR='${LOG_DIR}' '${ROOT}/scripts/run_gold_scenarios.sh' {}"
cluster_rc=$?
set -e

{
  echo "Done. Job log: ${LOG_DIR}/cluster_joblog.tsv"
  echo "Results under ${ROOT}/${BD_OUTDIR}/"
  echo "cluster_exit=${cluster_rc}"
} | tee -a "${LOG_DIR}/suite.log"

exit "${cluster_rc}"
