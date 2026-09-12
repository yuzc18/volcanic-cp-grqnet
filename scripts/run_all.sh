#!/usr/bin/env bash
# Extended synthetic reproducibility checks for the public repository.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo " Extended synthetic reproducibility checks"
echo "============================================================"
python scripts/generate_synthetic_data.py
python scripts/quick_test.py
python scripts/round2_component_smoke.py --erf-trees 4
python scripts/round3_protocol_smoke.py
python scripts/round4_baseline_smoke.py
python scripts/conformal_calibrate.py --smoke
python scripts/run_table7_calibration_designs.py --smoke
python scripts/run_round6_analysis.py --smoke
pytest -q

echo "============================================================"
echo " Extended checks complete."
echo " Synthetic numerical outputs are demonstration outputs only."
echo "============================================================"
