#!/bin/sh
# Supply the directory containing .fedpara-gui, not .fedpara-gui itself.
if [ "$#" -ne 1 ]; then
  echo 'Usage: sh scripts/start_kagami_with_previous_data.sh /path/to/old/working/directory' >&2
  exit 2
fi
export KAGAMI_DATA_DIR="$1"
exec python kagami.py
