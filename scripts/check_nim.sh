#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v nim >/dev/null 2>&1; then
  echo 'Nim compiler is not installed. Install Nim 2.x and run nimble install mummy.' >&2
  exit 127
fi
nim --version
nim check --threads:on --mm:orc --hints:off src/main.nim
nim c --threads:on --mm:orc -d:release -o:/tmp/fedpara-mummy-check src/main.nim
# The optional native backend needs installed libdatachannel and is NOT compiled here.
