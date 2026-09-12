#!/usr/bin/env bash
# Fast end-to-end synthetic demonstration of the manuscript data flow.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python scripts/release_audit.py
python scripts/generate_synthetic_data.py
python scripts/quick_test.py
python scripts/round2_component_smoke.py --erf-trees 2
python scripts/round3_protocol_smoke.py
python scripts/train_grqnet.py --smoke --output-dir outputs/virtual_demo/train
python scripts/conformal_calibrate.py --smoke --output-dir outputs/virtual_demo/conformal
python scripts/eval_blind.py --smoke --output-dir outputs/virtual_demo/blind

echo "Virtual-input end-to-end demo: PASS"
echo "Synthetic metrics are demonstration values and are not manuscript results."
