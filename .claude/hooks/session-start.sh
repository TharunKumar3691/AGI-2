#!/bin/bash
set -euo pipefail

# Only run in Claude Code on the web
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install the Kaggle CLI (idempotent)
pip install --quiet --upgrade kaggle

# Legacy credentials: build ~/.kaggle/kaggle.json from env secrets.
# New-style tokens (KAGGLE_API_TOKEN) are read by the CLI directly from the env.
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  mkdir -p "$HOME/.kaggle"
  umask 077
  printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$HOME/.kaggle/kaggle.json"
  chmod 600 "$HOME/.kaggle/kaggle.json"
elif [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "Kaggle credentials not set (KAGGLE_API_TOKEN or KAGGLE_USERNAME+KAGGLE_KEY); CLI installed but unauthenticated." >&2
fi
