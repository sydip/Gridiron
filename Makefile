# Gridiron local dashboard.
#
# Thin wrappers around run_local.sh, which holds the actual logic so that the
# Windows launcher (run_local.ps1) and this file cannot drift apart.
#
# NOTE: make is not installed on Windows by default. If `make dashboard`
# reports "command not found", use the equivalent directly:
#
#     PowerShell   .\run_local.ps1
#     Git Bash     ./run_local.sh
#
# Nothing here touches the pipeline. See LOCALHOST.md.

SHELL := /bin/bash
API_PORT ?= 8000
WEB_PORT ?= 3000

.PHONY: dashboard api web install stop help
.DEFAULT_GOAL := help

## dashboard: start the API and the frontend, and open a browser
dashboard:
	@./run_local.sh

## api: start only the backend on :8000
api:
	@./run_local.sh --api-only

## web: start only the frontend on :3000 (expects an API already running)
web:
	@./run_local.sh --web-only

## install: install the pip and npm dependencies the dashboard needs
install:
	@./run_local.sh --install

## stop: free ports 8000 and 3000 if a previous run was left behind
stop:
	@for port in $(API_PORT) $(WEB_PORT); do \
		pid=$$(netstat -ano 2>/dev/null | grep ":$$port " | grep LISTENING | awk '{print $$5}' | sort -u | head -1); \
		if [ -n "$$pid" ]; then \
			echo "  stopping pid $$pid on port $$port"; \
			taskkill //PID $$pid //F >/dev/null 2>&1 || kill -9 $$pid 2>/dev/null || true; \
		else \
			echo "  port $$port already free"; \
		fi; \
	done

help:
	@echo ""
	@echo "Gridiron local dashboard"
	@echo ""
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  make /'
	@echo ""
	@echo "  Without make:  .\\run_local.ps1  (PowerShell)  or  ./run_local.sh  (bash)"
	@echo ""
