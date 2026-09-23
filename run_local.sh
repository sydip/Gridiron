#!/usr/bin/env bash
# Start the Gridiron dashboard: API on :8000, frontend on :3000, browser open.
#
# Additive to the pipeline, not a replacement for it. This starts two dev
# servers that read what the pipeline has already produced; it never trains,
# predicts, or writes into outputs/ by itself.
#
#   ./run_local.sh              both servers, open a browser
#   ./run_local.sh --install    install pip and npm dependencies, then exit
#   ./run_local.sh --api-only   backend only
#   ./run_local.sh --web-only   frontend only
#   ./run_local.sh --no-browser don't open a browser
#
# Ctrl+C stops both.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT=8000
WEB_PORT=3000

API_ONLY=0
WEB_ONLY=0
NO_BROWSER=0
INSTALL=0

for arg in "$@"; do
  case "$arg" in
    --api-only) API_ONLY=1 ;;
    --web-only) WEB_ONLY=1 ;;
    --no-browser) NO_BROWSER=1 ;;
    --install) INSTALL=1 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '  %s\n' "$1"; }
bad() { printf '  %s\n' "$1" >&2; }

# The project virtualenv is the one with gridiron installed; prefer it.
project_python() {
  for candidate in "$PROJECT_ROOT/.venv/Scripts/python.exe" \
                   "$PROJECT_ROOT/.venv/bin/python"; do
    [ -x "$candidate" ] && { echo "$candidate"; return; }
  done
  command -v python3 || command -v python || {
    bad "No Python found. Create the virtualenv first: python -m venv .venv"
    exit 1
  }
}

port_busy() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$1" -sTCP:LISTEN -t >/dev/null 2>&1
  else
    netstat -ano 2>/dev/null | grep -q ":$1 .*LISTENING"
  fi
}

assert_port_free() {
  if port_busy "$1"; then
    bad "Port $1 is already in use, so $2 cannot start there."
    exit 1
  fi
}

open_browser() {
  local url="$1"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then open "$url" >/dev/null 2>&1 &
  elif command -v start >/dev/null 2>&1; then start "$url" >/dev/null 2>&1 &
  else step "Open $url in a browser."
  fi
}

PYTHON="$(project_python)"

if [ "$INSTALL" -eq 1 ]; then
  printf '\nInstalling dependencies\n'
  step "pip: api/requirements.txt"
  "$PYTHON" -m pip install -q -r "$PROJECT_ROOT/api/requirements.txt"
  step "npm: web/"
  (cd "$PROJECT_ROOT/web" && npm install --silent)
  step "Done. Now run: ./run_local.sh"
  exit 0
fi

printf '\nGridiron dashboard\n'

[ "$WEB_ONLY" -eq 1 ] || assert_port_free "$API_PORT" "the API"
if [ "$API_ONLY" -eq 0 ]; then
  assert_port_free "$WEB_PORT" "the frontend"
  if [ ! -d "$PROJECT_ROOT/web/node_modules" ]; then
    bad "web/node_modules is missing. Run: ./run_local.sh --install"
    exit 1
  fi
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

if [ "$WEB_ONLY" -eq 0 ]; then
  step "Starting API on http://localhost:$API_PORT"
  (cd "$PROJECT_ROOT" && "$PYTHON" -m uvicorn api.main:app --reload --port "$API_PORT") &
  PIDS+=($!)

  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1; then break; fi
    sleep 0.5
  done
  if ! curl -fsS "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1; then
    bad "The API did not answer /api/health within 45 seconds."
    exit 1
  fi

  # A degraded API still serves; it just has artifacts missing. Name them,
  # rather than letting the page show empty panels for no stated reason.
  status="$(curl -fsS "http://127.0.0.1:$API_PORT/api/health" \
    | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
  if [ "$status" = "ok" ]; then
    step "API ready (all artifacts present)"
  else
    bad "API is running but some artifacts are missing:"
    curl -fsS "http://127.0.0.1:$API_PORT/api/health" | "$PYTHON" -c '
import json, sys
for name, a in json.load(sys.stdin)["artifacts"].items():
    if not a["present"]:
        print(f"    {a[\"path\"]}  ->  {a[\"remedy\"]}")
'
  fi
fi

if [ "$API_ONLY" -eq 1 ]; then
  printf '\n  API only. Ctrl+C to stop.\n\n'
  wait
  exit 0
fi

step "Starting frontend on http://localhost:$WEB_PORT"
(cd "$PROJECT_ROOT/web" && npm run dev) &
PIDS+=($!)

for _ in $(seq 1 90); do
  port_busy "$WEB_PORT" && break
  sleep 0.5
done
if ! port_busy "$WEB_PORT"; then
  bad "The frontend did not come up on port $WEB_PORT within 45 seconds."
  exit 1
fi

[ "$NO_BROWSER" -eq 1 ] || { step "Opening the browser"; open_browser "http://localhost:$WEB_PORT"; }

printf '\n  Dashboard  http://localhost:%s\n' "$WEB_PORT"
printf '  API docs   http://localhost:%s/docs\n' "$API_PORT"
printf '  Ctrl+C to stop both.\n\n'

wait
