#!/usr/bin/env bash
# Backward-compatible wrapper for the full paper reproduction pipeline.
#
# The maintained driver is scripts/arXiv_recreate.sh. Keep this file so older
# notes or shell history that call paper_recreate.sh still reproduce the full
# arXiv result set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[paper_recreate.sh] Deprecated wrapper; running scripts/arXiv_recreate.sh" >&2
exec bash "${SCRIPT_DIR}/arXiv_recreate.sh" "$@"
