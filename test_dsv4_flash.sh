#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/test_dsv4.sh" flash "$@"
