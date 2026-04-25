#!/usr/bin/env bash

set -euo pipefail

CONFIG="configs/data.yaml"
PYTHON_BIN="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        exit 1
      fi
      CONFIG="$2"
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --python" >&2
        exit 1
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/preprocess.sh [--config PATH] [--python PYTHON]

Runs the full preprocessing pipeline in the required Spark order:
  build_cohort -> stage_filtered_tables -> build_vocab -> build_drugbank_metadata ->
  build_ddi_matrix -> build_drugbank_ddi_matrix -> build_trajectories
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Use --help for usage." >&2
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m src.data.build_cohort --config "$CONFIG"
"$PYTHON_BIN" -m src.data.stage_filtered_tables --config "$CONFIG"
"$PYTHON_BIN" -m src.data.build_vocab --config "$CONFIG"
"$PYTHON_BIN" -m src.data.build_drugbank_metadata --config "$CONFIG"
"$PYTHON_BIN" -m src.data.build_ddi_matrix --config "$CONFIG"
"$PYTHON_BIN" -m src.data.build_drugbank_ddi_matrix --config "$CONFIG"
"$PYTHON_BIN" -m src.data.build_trajectories --config "$CONFIG"
