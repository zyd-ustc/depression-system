#!/bin/bash
# Run the Gradio UI.

set -euo pipefail

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

# PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Run
python src/inference_pipeline/ui.py


