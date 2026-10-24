#!/usr/bin/env bash
# Run ONLY custom/factory model training or importing in an offline sandbox.
# Example: sandbox/run_untrusted.sh /path/to/repo /path/to/batches /path/to/out \
#          train --model /model --model-id org/name --revision FULL40HEX \
#          --trust-remote-code --batches-dir /data --peer-id alice --output /out/a.fpatch
set -euo pipefail
if [[ "$#" -lt 4 ]]; then
  echo "Usage: $0 ABS_MODEL_DIR ABS_BATCH_DIR ABS_OUTPUT_DIR CLI_COMMAND [args...]" >&2
  exit 2
fi
model_dir=$(cd -- "$1" && pwd -P)
data_dir=$(cd -- "$2" && pwd -P)
output_dir=$(cd -- "$3" && pwd -P)
shift 3
if [[ ! -d "$model_dir" || ! -d "$data_dir" || ! -d "$output_dir" ]]; then
  echo 'Model, data and output directories must already exist.' >&2; exit 2
fi
case "$1" in
  train|state-export|state-import|merge) ;;
  *) echo 'Only offline CLI operations are allowed in this sandbox.' >&2; exit 2 ;;
esac
# Build offline-runtime image separately: docker build -f sandbox/Dockerfile.offline -t fedpara-untrusted-cpu .
# Disable networking and privileges, mount only explicitly chosen directories.
exec docker run --rm -i \
  --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 128 --memory 8g --cpus 4 \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 \
  -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_MODULES_CACHE=/tmp/hf/modules \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e FEDPARA_ISOLATED_RUNTIME=1 \
  -v "$model_dir:/model:ro" \
  -v "$data_dir:/data:ro" -v "$output_dir:/out:rw" \
  -w /app fedpara-untrusted-cpu python -m patchlib.cli "$@"
