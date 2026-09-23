<#
.SYNOPSIS
    Start the Gridiron dashboard: API on :8000, frontend on :3000, browser open.

.DESCRIPTION
    Additive to the pipeline, not a replacement for it. This starts two dev
    servers that read what the pipeline has already produced; it never trains,
    predicts or writes into outputs/ by itself.

    Both servers run in this window. Ctrl+C stops them together.

.PARAMETER ApiOnly
    Start only the backend.

.PARAMETER WebOnly
    Start only the frontend. Assumes an API is already listening on :8000.

.PARAMETER NoBrowser
    Do not open a browser window.

.PARAMETER Install
    Install/refresh dependencies (pip for the API, npm for the frontend), then exit.

.EXAMPLE
    .\run_local.ps1
    .\run_local.ps1 -Install
    .\run_local.ps1 -NoBrowser
#>
[CmdletBinding()]
param(
    [switch]$ApiOnly,
    [switch]$WebOnly,
    [switch]$NoBrowser,
    [switch]$Install
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$ApiPort = 8000
$WebPort = 3000

function Write-Step($message) { Write-Host "  $message" -ForegroundColor Cyan }
function Write-Bad($message) { Write-Host "  $message" -ForegroundColor Red }

function Get-ProjectPython {
    # Prefer the project's own virtualenv; it is the one with gridiron installed.
    $venv = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (Test-Path $venv) { return $venv }
    $found = Get-Command python -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    throw "No Python found. Create the virtualenv first: python -m venv .venv"
}

function Test-PortBusy($port) {
    $connections = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    return $null -ne $connections
}

function Assert-PortFree($port, $name) {
    if (Test-PortBusy $port) {
        Write-Bad "Port $port is already in use, so $name cannot start there."
        Write-Bad "Find it with:  Get-NetTCPConnection -State Listen -LocalPort $port"
        throw "Port $port busy"
    }
}

function Wait-ForApi($timeoutSeconds = 45) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 2
            return $response
        } catch {
            Start-Sleep -Milliseconds 400
        }
    }
    return $null
}

$python = Get-ProjectPython

# --- install ---------------------------------------------------------------

if ($Install) {
    Write-Host "`nInstalling dependencies" -ForegroundColor White
    Write-Step "pip: api/requirements.txt"
    & $python -m pip install -q -r (Join-Path $ProjectRoot 'api\requirements.txt')
    Write-Step "npm: web/"
    Push-Location (Join-Path $ProjectRoot 'web')
    try { npm install --silent } finally { Pop-Location }
    Write-Host "  Done. Now run: .\run_local.ps1" -ForegroundColor Green
    return
}

# --- preflight -------------------------------------------------------------

Write-Host "`nGridiron dashboard" -ForegroundColor White

if (-not $WebOnly) { Assert-PortFree $ApiPort 'the API' }
if (-not $ApiOnly) {
    Assert-PortFree $WebPort 'the frontend'
    if (-not (Test-Path (Join-Path $ProjectRoot 'web\node_modules'))) {
        Write-Bad "web/node_modules is missing. Run:  .\run_local.ps1 -Install"
        throw "Frontend dependencies not installed"
    }
}

$jobs = @()

try {
    # --- backend -----------------------------------------------------------

    if (-not $WebOnly) {
        Write-Step "Starting API on http://localhost:$ApiPort"
        $jobs += Start-Process -FilePath $python `
            -ArgumentList '-m', 'uvicorn', 'api.main:app', '--reload', '--port', $ApiPort `
            -WorkingDirectory $ProjectRoot -PassThru -NoNewWindow

        $health = Wait-ForApi
        if ($null -eq $health) {
            Write-Bad "The API did not answer /api/health within 45 seconds."
            throw "API failed to start"
        }

        # A degraded API still serves; it just has artifacts missing. Say which,
        # rather than letting the page show empty panels for no stated reason.
        if ($health.status -ne 'ok') {
            Write-Bad "API is running but some artifacts are missing:"
            foreach ($name in $health.artifacts.PSObject.Properties.Name) {
                $artifact = $health.artifacts.$name
                if (-not $artifact.present) {
                    Write-Bad "    $($artifact.path)  ->  $($artifact.remedy)"
                }
            }
        } else {
            Write-Step "API ready (all artifacts present)"
        }
    }

    if ($ApiOnly) {
        Write-Host "`n  API only. Ctrl+C to stop.`n" -ForegroundColor Green
        Wait-Process -Id $jobs[0].Id
        return
    }

    # --- frontend ----------------------------------------------------------

    Write-Step "Starting frontend on http://localhost:$WebPort"
    $jobs += Start-Process -FilePath 'npm.cmd' -ArgumentList 'run', 'dev' `
        -WorkingDirectory (Join-Path $ProjectRoot 'web') -PassThru -NoNewWindow

    $deadline = (Get-Date).AddSeconds(45)
    while ((Get-Date) -lt $deadline -and -not (Test-PortBusy $WebPort)) {
        Start-Sleep -Milliseconds 400
    }
    if (-not (Test-PortBusy $WebPort)) {
        Write-Bad "The frontend did not come up on port $WebPort within 45 seconds."
        throw "Frontend failed to start"
    }

    if (-not $NoBrowser) {
        Write-Step "Opening the browser"
        Start-Process "http://localhost:$WebPort"
    }

    Write-Host ""
    Write-Host "  Dashboard  http://localhost:$WebPort" -ForegroundColor Green
    Write-Host "  API docs   http://localhost:$ApiPort/docs" -ForegroundColor Green
    Write-Host "  Ctrl+C to stop both.`n" -ForegroundColor Green

    while ($true) {
        Start-Sleep -Seconds 1
        foreach ($job in $jobs) {
            if ($job.HasExited) {
                Write-Bad "A server exited (pid $($job.Id), code $($job.ExitCode)). Shutting down."
                return
            }
        }
    }
} finally {
    # Runs on Ctrl+C too, so neither server is left holding a port.
    foreach ($job in $jobs) {
        if ($job -and -not $job.HasExited) {
            Write-Host "  Stopping pid $($job.Id)" -ForegroundColor DarkGray
            Stop-Process -Id $job.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
