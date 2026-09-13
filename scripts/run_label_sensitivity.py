#!/usr/bin/env python3
"""Run the manuscript Table-8 label-definition sensitivity workflow.

Table 8(a) scans c=2..6 on the 890 development rows. Table 8(b) refits the
quality classifier for the adopted c=3 K-means labels and the c=2, c=4, and
Manhattan K-medoids alternatives, then recalibrates randomized APS.

The active main-analysis 178-row calibration assignment is held fixed across
label definitions and only its labels change. A study-specific split file can be
passed through ``--split-metadata`` when a private row mapping is available.

The released LightGBM HPO space is machine-readable in
``configs/search_spaces.yaml``. ``--reported-fixed-demo`` remains a fast path
that uses the reported Table-3(a) LightGBM configuration directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.analysis.label_sensitivity import (  # noqa: E402
    cluster_validity_scan,
    kmeans_definition,
    kmedoids_definition,
)
from src.config import load_wells_config  # noqa: E402
from src.data.splits import FullPartition  # noqa: E402
from src.models.grq_net import GRQNetConfig  # noqa: E402
from src.pipeline import (  # noqa: E402
    ExperimentContext,
    RuntimeOptions,
    build_context,
    prepare_final_model,
    prepare_outer_fold,
)
from src.training.baseline_non_neural import (  # noqa: E402
    build_non_neural_estimator,
    fixed_reported_params,
    load_search_protocol,
    tune_non_neural,
)
from src.training.trainer import TrainConfig, predict_grqnet, train_one_fold  # noqa: E402
from src.uncertainty.conformal import (  # noqa: E402
    aps_nonconformity_scores,
    build_prediction_sets_detailed,
    compute_threshold,
)


def _opts(smoke: bool, seed: int) -> RuntimeOptions:
    if not smoke:
        return RuntimeOptions(seed=seed)
    return RuntimeOptions(
        seed=seed, erf_n_estimators=1, erf_max_depth=4,
        xgb_n_estimators=2, max_lith_rows_per_class=4, max_epochs=1,
    )


def _context_with_labels(base: ExperimentContext, labels_all: np.ndarray) -> ExperimentContext:
    labeled = base.labeled_core.copy()
    if len(labels_all) != len(labeled):
        raise ValueError("Alternative labels must cover all labeled core rows.")
    labeled["quality_class_id"] = np.asarray(labels_all, dtype=int)

    # Table-8 sensitivity keeps the already selected calibration row identities
    # fixed and changes only their labels. Re-running the main-label quota validator
    # would be incorrect for c=2/c=4 alternatives because those rows no longer have
    # the 51/49/78 three-class composition by construction.
    label_by_id = labeled.set_index("sample_id")["quality_class_id"]
    dev = base.development.copy()
    blind = base.blind.copy()
    dev["quality_class_id"] = dev["sample_id"].map(label_by_id).to_numpy(dtype=int)
    blind["quality_class_id"] = blind["sample_id"].map(label_by_id).to_numpy(dtype=int)
    partition = FullPartition(
        development_df=dev,
        blind_df=blind,
        calibration=base.partition.calibration,
        folds=base.partition.folds,
    )
    return ExperimentContext(
        inputs=base.inputs,
        wells=base.wells,
        labeler=base.labeler,
        labeled_core=labeled,
        partition=partition,
        development_labels=dev["quality_class_id"].to_numpy(dtype=int),
        calibration_sample_ids=base.calibration_sample_ids,
    )


def _grqnet_and_cp(ctx: ExperimentContext, *, n_classes: int, options: RuntimeOptions, alpha: float, seed: int):
    fold_f1 = []
    rng = np.random.default_rng(seed)
    for fold in ctx.partition.folds:
        p = prepare_outer_fold(ctx, fold, options=options)
        g, e = p.gradient_local_idx, p.early_stop_local_idx
        cfg = TrainConfig(max_epochs=options.max_epochs, n_classes=n_classes)
        model_cfg = GRQNetConfig(n_classes=n_classes)
        trained = train_one_fold(
            p.X_outer_train[g], p.y_outer_train[g],
            p.X_outer_train[e], p.y_outer_train[e],
            train_cfg=cfg, model_cfg=model_cfg, seed=seed,
            class_weight_labels=p.y_outer_train,
        )
        pred = predict_grqnet(trained.model, p.X_outer_val)
        fold_f1.append(float(f1_score(p.y_outer_val, pred.predictions, average="macro")))

    final_p = prepare_final_model(ctx, options=options)
    g, e = final_p.gradient_local_idx, final_p.early_stop_local_idx
    cfg = TrainConfig(max_epochs=options.max_epochs, n_classes=n_classes)
    model_cfg = GRQNetConfig(n_classes=n_classes)
    final_tr = train_one_fold(
        final_p.X_modeling[g], final_p.y_modeling[g],
        final_p.X_modeling[e], final_p.y_modeling[e],
        train_cfg=cfg, model_cfg=model_cfg, seed=seed,
        class_weight_labels=final_p.y_modeling,
    )
    p_cal = predict_grqnet(final_tr.model, final_p.X_calibration).probabilities
    p_blind = predict_grqnet(final_tr.model, final_p.X_blind).probabilities
    u_cal = rng.uniform(0, 1, len(p_cal))
    scores = aps_nonconformity_scores(p_cal, final_p.y_calibration, uniforms=u_cal)
    qhat = compute_threshold(scores, alpha)
    u_blind = rng.uniform(0, 1, len(p_blind))
    sets = build_prediction_sets_detailed(p_blind, qhat, uniforms=u_blind)
    covered = np.array([int(y) in s for y, s in zip(final_p.y_blind, sets.sets, strict=True)])
    return float(np.mean(fold_f1)), float(np.std(fold_f1, ddof=1)), float(covered.mean()), qhat


def _gbm_cv(
    ctx: ExperimentContext,
    *,
    n_classes: int,
    options: RuntimeOptions,
    seed: int,
    smoke: bool,
    reported_fixed_demo: bool,
) -> tuple[float, float, str, list[dict]]:
    """Table-8 LightGBM comparison under one alternative label definition.

    P102 selects LightGBM hyperparameters inside each outer training fold, and
    P160 re-runs Table 8 under the same five-fold protocol.  The default path
    therefore calls ``tune_non_neural`` for every (label definition, outer fold)
    pair, selecting on that fold's internal 15% split with the alternative
    labels, then refits the selected configuration on the whole outer-training
    subset -- the same protocol ``run_baselines.py`` uses for Table 6(a).

    ``reported_fixed_demo`` is a separate fixed-configuration demonstration
    branch.  It is never reported as HPO: it returns its own selection-engine
    label and no selected-parameter records.

    Returns the fold mean, the fold standard deviation, the selection-engine
    label, and one record per fold.
    """
    cfg = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    vals: list[float] = []
    records: list[dict] = []

    if reported_fixed_demo:
        params = fixed_reported_params("lightgbm", cfg, smoke=smoke)
        for fold in ctx.partition.folds:
            p = prepare_outer_fold(ctx, fold, options=options)
            m = build_non_neural_estimator("lightgbm", params, seed=seed, n_classes=n_classes)
            m.fit(p.X_outer_train, p.y_outer_train)
            vals.append(float(f1_score(p.y_outer_val, m.predict(p.X_outer_val), average="macro")))
            records.append({
                "fold": int(fold.fold_id),
                "validation_well": fold.validation_well,
                "selection_engine": "reported_table3a_fixed_demo_not_hpo",
                "internal_macro_f1": None,
                "selected_params": json.dumps(params, sort_keys=True),
            })
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1)),
            "reported_table3a_fixed_demo_not_hpo",
            records,
        )

    protocol = load_search_protocol(ROOT / "configs" / "search_spaces.yaml")
    engines: set[str] = set()
    for fold in ctx.partition.folds:
        p = prepare_outer_fold(ctx, fold, options=options)
        g, e = p.gradient_local_idx, p.early_stop_local_idx
        sel = tune_non_neural(
            "lightgbm",
            p.X_outer_train[g], p.y_outer_train[g],
            p.X_outer_train[e], p.y_outer_train[e],
            protocol=protocol,
            baseline_cfg=cfg,
            seed=seed,
            refit_X=p.X_outer_train,
            refit_y=p.y_outer_train,
            smoke=smoke,
            n_classes=n_classes,
        )
        engines.add(sel.engine)
        vals.append(
            float(f1_score(p.y_outer_val, sel.model.predict(p.X_outer_val), average="macro"))
        )
        records.append({
            "fold": int(fold.fold_id),
            "validation_well": fold.validation_well,
            "selection_engine": sel.engine,
            "internal_macro_f1": float(sel.internal_macro_f1),
            "selected_params": json.dumps(sel.best_params, sort_keys=True),
        })
    engine = engines.pop() if len(engines) == 1 else "+".join(sorted(engines))
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), engine, records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "table8")
    ap.add_argument("--split-metadata", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reported-fixed-demo", action="store_true")
    args = ap.parse_args()

    if not (args.data_dir / "continuous_logs.csv").exists():
        if args.data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(args.data_dir)
        write_synthetic_inputs(args.data_dir, args.seed)

    base = build_context(args.data_dir, cp_seed=args.seed, split_metadata_path=args.split_metadata)
    full = base.labeled_core.copy()
    devmask = full["well_id"].isin(load_wells_config().training_well_names).to_numpy()
    scan = cluster_validity_scan(full.loc[devmask])

    adopted = full["quality_class_id"].to_numpy(dtype=int)
    defs = [
        ("kmeans_c3_adopted", 3, adopted, np.asarray(base.labeler.boundaries_.as_array()) if base.labeler.boundaries_ else np.array([])),
    ]
    for c in (2, 4):
        d = kmeans_definition(full, devmask, n_classes=c)
        defs.append((d.name, d.n_classes, d.labels_all, d.boundaries))
    try:
        d = kmedoids_definition(full, devmask)
        defs.append((d.name, d.n_classes, d.labels_all, d.boundaries))
    except ImportError:
        if not args.smoke:
            raise
        print("K-medoids smoke row skipped because scikit-learn-extra is not installed in this runner.")

    options = _opts(args.smoke, args.seed)
    rows = []
    gbm_config_records: list[dict] = []
    for name, k, labels_all, boundaries in defs:
        ctx = _context_with_labels(base, labels_all)
        grq_mean, grq_sd, coverage, qhat = _grqnet_and_cp(
            ctx, n_classes=k, options=options, alpha=args.alpha, seed=args.seed
        )
        gbm_mean, gbm_sd, gbm_engine, gbm_records = _gbm_cv(
            ctx, n_classes=k, options=options, seed=args.seed, smoke=args.smoke,
            reported_fixed_demo=args.reported_fixed_demo,
        )
        for rec in gbm_records:
            gbm_config_records.append({"label_definition": name, "n_classes": k, **rec})
        rows.append({
            "label_definition": name,
            "n_classes": k,
            "fzi_boundaries_um": " / ".join(f"{x:.6g}" for x in boundaries),
            # P160: the ARI is computed on the 890 development samples used for
            # label construction, not on all 1259 core-matched rows.
            "ari_vs_adopted": float(adjusted_rand_score(adopted[devmask], labels_all[devmask])),
            "n_ari_samples": int(devmask.sum()),
            "grqnet_macro_f1_mean": grq_mean,
            "grqnet_macro_f1_sd": grq_sd,
            "gbm_macro_f1_mean": gbm_mean,
            "gbm_macro_f1_sd": gbm_sd,
            "grqnet_minus_gbm_pp": 100.0 * (grq_mean - gbm_mean),
            "blind_coverage": coverage,
            "deployment_qhat": qhat,
            "gbm_selection_engine": gbm_engine,
            "gbm_configuration_source": (
                "Table-3(a) fixed configuration; not HPO"
                if args.reported_fixed_demo
                else "per-outer-fold selection on the internal 15% split, under this label definition"
            ),
        })
        print(f"{name}: GRQ={grq_mean:.3f} GBM={gbm_mean:.3f} coverage={coverage:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan.to_csv(args.output_dir / "table8a_cluster_validity.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "table8b_alternative_labels.csv", index=False)
    # Auditable record of what was actually selected for every
    # (label definition, outer fold) pair.
    pd.DataFrame(gbm_config_records).to_csv(
        args.output_dir / "table8b_gbm_per_fold_configurations.csv", index=False
    )
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "seed": args.seed,
        "alpha": args.alpha,
        "smoke": args.smoke,
        "calibration_row_policy": "same active 178-row assignment across alternative label definitions; labels recomputed",
        "gbm_reported_fixed_demo": bool(args.reported_fixed_demo),
        "gbm_selection": (
            "fixed Table-3(a) configuration, not hyperparameter optimization"
            if args.reported_fixed_demo
            else "tune_non_neural per label definition and outer fold; "
                 "selection on the internal 15% split; refit on the outer-training subset"
        ),
        "ari_population": "890 development samples used for label construction (P160)",
        "private_study_data_included": False,
    }, indent=2), encoding="utf-8")
    print("Table-8 label-sensitivity run: PASS")


if __name__ == "__main__":
    main()
