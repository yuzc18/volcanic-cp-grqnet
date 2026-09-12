"""Round-2 component smoke test on synthetic inputs.

This is *not* the manuscript cross-validation driver.  It verifies that the
reconstructed Round-2 components can execute together without proprietary data:
FZI labels -> shared 3σ clipping -> continuous-grid Group D -> Group-B eRF ->
Group-C XGBoost -> scaled 50-dimensional UFT.

Outer/inner leave-one-well-out cross-fitting is intentionally added in Round 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.config import load_wells_config  # noqa: E402
from src.data.io import load_inputs, match_core_to_continuous  # noqa: E402
from src.data.labels import fit_development_labels  # noqa: E402
from src.features.group_a import RawLogClipper  # noqa: E402
from src.features.group_b import ERFLithologyClassifier  # noqa: E402
from src.features.group_c import GroupCRegressors  # noqa: E402
from src.features.group_d import build_group_d_continuous, match_group_d_to_core  # noqa: E402
from src.features.uft import UFTPreprocessor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--erf-trees", type=int, default=6, help="Reduced smoke-test ensemble; paper configuration is 300.")
    args = p.parse_args()

    if not (args.data_dir / "continuous_logs.csv").exists():
        write_synthetic_inputs(args.data_dir, args.seed)

    inputs = load_inputs(args.data_dir)
    wells = load_wells_config()
    core = match_core_to_continuous(inputs.core_samples, inputs.continuous_logs)
    labeler, labeled = fit_development_labels(core, wells.training_well_names)

    # Component-level holdout used only to ensure fit/transform separation in
    # this smoke test. Round 3 replaces this with the exact paper split logic.
    demo_holdout = wells.training_well_names[0]
    fit_mask = labeled["well_id"].isin(wells.training_well_names[1:]).to_numpy()

    clipper = RawLogClipper().fit(labeled.loc[fit_mask])
    d_cont = build_group_d_continuous(inputs.continuous_logs, clipper=clipper, causal=False)
    d_core = match_group_d_to_core(labeled, d_cont)

    lith_fit = inputs.lithology_corpus.loc[
        inputs.lithology_corpus["well_id"].isin(wells.training_well_names[1:])
    ].copy()
    # Keep the component smoke test fast without changing the paper-facing
    # 300-tree configuration. This deterministic class-balanced subsample is
    # synthetic only and is never used by research-mode orchestration.
    lith_fit = (
        lith_fit.groupby("lithology_class_id", group_keys=False)
        .head(20)
        .reset_index(drop=True)
    )
    erf = ERFLithologyClassifier(
        n_estimators=args.erf_trees, random_state=42, max_depth=12
    ).fit(lith_fit)
    B = erf.predict_frame(labeled)

    group_c = GroupCRegressors(random_state=args.seed).fit(labeled.loc[fit_mask])
    C = group_c.predict(labeled)

    pre = UFTPreprocessor().fit_on(
        labeled.loc[fit_mask],
        B.loc[fit_mask],
        C.as_array()[fit_mask],
        d_core.loc[fit_mask],
        raw_clipper=clipper,
    )
    X = pre.transform(labeled, B, C, d_core)

    assert X.shape == (1259, 50)
    assert np.isfinite(X).all()
    assert np.allclose(B.to_numpy().sum(axis=1), 1.0, atol=1e-6)
    print("Round-2 component smoke test: PASS")
    print(f"  synthetic core rows: {len(labeled)}")
    print(f"  FZI boundaries from synthetic development data: {labeler.boundaries_.as_array().tolist()}")
    print(f"  demonstration fit wells: {wells.training_well_names[1:]}")
    print(f"  demonstration held-out well: {demo_holdout}")
    print(f"  Group B shape: {B.shape}; smoke-test eRF trees: {args.erf_trees}")
    print(f"  Group C shape: {C.as_array().shape}")
    print(f"  UFT shape: {X.shape}")
    print("  NOTE: synthetic outputs are not manuscript-result reproductions.")


if __name__ == "__main__":
    main()
