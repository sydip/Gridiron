# Running the Gridiron dashboard locally

A browser view of what the pipeline has already produced. Two dev servers: a
read-only API on **:8000** and a frontend on **:3000**.

This is additive. It does not train, predict, or write into `outputs/` by
itself, and nothing in `src/gridiron/`, `scripts/`, `tests/`, `configs/`,
`data/` or `models/` was changed to add it. The pipeline is documented in
[README.md](README.md); this file covers only the dashboard.

## First run

```powershell
cd C:\Users\saide\.vscode\Gridiron
.\run_local.ps1 -Install     # pip for the API, npm for the frontend; once only
.\run_local.ps1              # starts both, opens the browser
```

That's it. The dashboard is at <http://localhost:3000> and the API's
interactive docs at <http://localhost:8000/docs>. Ctrl+C stops both.

### The other ways in

| Shell | Command |
| --- | --- |
| PowerShell (Windows) | `.\run_local.ps1` |
| Git Bash / macOS / Linux | `./run_local.sh` |
| make | `make dashboard` |

All three run the same thing. **`make` is not installed on Windows by
default** — if `make dashboard` reports "command not found", use
`.\run_local.ps1`. The Makefile delegates to `run_local.sh` so the three
cannot drift apart.

### Options

| Flag (PowerShell / bash) | Effect |
| --- | --- |
| `-Install` / `--install` | Install dependencies, then exit |
| `-ApiOnly` / `--api-only` | Backend only, on :8000 |
| `-WebOnly` / `--web-only` | Frontend only; expects an API already on :8000 |
| `-NoBrowser` / `--no-browser` | Don't open a browser |

`make api`, `make web`, `make install` and `make stop` map to the same things.

## What you need first

The dashboard **reads** pipeline outputs — it cannot generate them. On a fresh
clone with nothing built, it starts but has nothing to show.

```powershell
python -m gridiron.cli all          # builds everything the dashboard reads
python -m gridiron.cli predict --season 2026 --week 3
```

If artifacts are missing, the launcher says which, and names the command that
produces each one, before the browser opens:

```text
  API is running but some artifacts are missing:
    outputs/reports/walk_forward_aggregate.csv  ->  python -m gridiron.cli backtest
```

<http://localhost:8000/api/health> reports the same thing at any time.

## How it fits with the pipeline commands

The two are independent. Run pipeline commands in one terminal and leave the
dashboard running in another; the API reads from disk on every request, so a
refresh of the browser picks up whatever the pipeline last wrote. Nothing needs
restarting.

| Terminal 1 (pipeline) | Terminal 2 (dashboard) |
| --- | --- |
| `python -m gridiron.cli backtest` | leave running |
| `python -m gridiron.cli predict --season 2026 --week 4` | reload the page |

There is also `POST /api/refresh/{week}`, which runs
`python -m gridiron.cli predict --season 2026 --week N` as a subprocess so a
week can be generated without leaving the browser. It is the same command with
the same output — it adds no prediction logic of its own. It looks for an
interpreter that can import `gridiron`: `$GRIDIRON_PYTHON` first, then the
project's `.venv`, then its own.

## What the dashboard can and cannot show

**Completed weeks have no predictions, and cannot be given any.**
`predict_week` refuses games that have already been played, because a finished
game has a result rather than a recommendation. 2026 weeks 1 and 2 finished
before the model was pointed at them, so they show their results with no pick
against them. `POST /api/refresh/1` returns a nonzero exit code and the
command's own refusal. This is the pipeline behaving correctly, not a gap in
the dashboard.

**Weeks without published lines show as pending.** nflverse posts spreads a
week or two ahead, so later weeks list their matchups with no line and no
probability until the lines appear.

**The numbers are not re-derived here.** Every value the API serves is read
from a file the pipeline wrote, at full precision, and every response names the
files it came from. If a figure looks wrong, it is wrong in the CSV, and the
fix belongs in the pipeline.

## Ports

Both are fixed, because this document and the launcher promise those
addresses. A clash fails loudly rather than moving the app somewhere nobody is
looking:

```text
  Port 3000 is already in use, so the frontend cannot start there.
```

To clear a port left behind by an earlier run:

```powershell
Get-NetTCPConnection -State Listen -LocalPort 8000 |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

or `make stop`, which clears both.

The browser talks only to :3000; Vite proxies `/api` through to :8000, so the
page stays same-origin and never depends on CORS. The API also sets CORS
headers for any `localhost` port, for anyone serving the frontend differently.

## Layout

```text
api/            FastAPI service (new)
  main.py       routes
  sources.py    the only place files are opened
  README.md     endpoint list and the field -> file mapping
web/            React + TypeScript frontend (new)
  src/          components, pages, api client, design tokens
  vite.config.ts
  README.md     design system and routes
run_local.ps1   launcher (Windows)
run_local.sh    launcher (bash)
Makefile        thin wrapper over run_local.sh
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `make: command not found` | make isn't installed on Windows. Use `.\run_local.ps1`. |
| `web/node_modules is missing` | Run `.\run_local.ps1 -Install`. |
| `Port 8000 is already in use` | An earlier run is still up. `make stop`, or stop the process by PID. |
| Page says "Could not reach the API" | The backend didn't start. Run `.\run_local.ps1 -ApiOnly` to see its errors. |
| Panels are empty, health says `degraded` | Artifacts not generated. `python -m gridiron.cli all`. |
| Week shows no predictions | Either the week is already played (expected), or it has no lines yet. |
| `run_local.ps1 cannot be loaded` | Execution policy. `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
