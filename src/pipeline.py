"""Main orchestration through GRQ-Net fitting for the finalized manuscript.

Round 3 replaces the disabled pre-revision driver with the manuscript-faithful
split and cross-fitting protocol.  Conformal prediction is deliberately not
performed here; Round 5 consumes the fold-specific and final calibration
probabilities produced by this module.

The public pipeline is fully runnable on the repository's explicitly synthetic
inputs.  Numerical outputs from synthetic data are demonstration outputs only
and are not expected to reproduce manuscript results.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import WellsConfig, load_wells_config
from src.data.io import StudyInputs, load_inputs, match_core_to_continuous
from src.data.labels import FZILabeler, fit_development_labels
from src.data.splits import (
    CVFold,
    FullPartition,
    build_full_partition,
    CalibrationSplit,
    make_early_stopping_split,
    make_final_early_stopping_split,
    make_outer_early_stopping_split,
)
from src.features.group_a import RawLogClipper
from src.features.group_d import build_group_d_continuous, match_group_d_to_core
from src.features.uft import UFTPreprocessor
from src.training.metrics import ClassificationMetrics, compute_metrics
from src.training.trainer import (
    FoldResult,
    PredictionResult,
    TrainConfig,
    predict_grqnet,
    train_one_fold,
)
from src.training.upstream import (
    ProxyBundle,
    UpstreamAudit,
    assert_crossfit_audits_no_target_well,
    crossfit_training_proxies,
    fit_evaluation_upstream,
)


@dataclass(frozen=True)
class RuntimeOptions:
    """Execution controls.

    Defaults are paper-facing.  ``--smoke`` in the command-line driver changes
    only these execution controls for fast synthetic CI; it does not change the
    canonical YAML manuscript configuration.
    """

    seed: int = 20260827
    erf_random_state: int = 42
    erf_n_estimators: int = 300
    erf_max_depth: int | None = None
    xgb_n_estimators: int = 300
    max_lith_rows_per_class: int | None = None
    max_epochs: int = 200
    #: Group D window definition, ``"centered"`` (Table 2(b), used for every
    #: headline result) or ``"causal"`` (the strictly causal variant of
    #: Section 3.3, which needs no future log sample).  Selecting ``"causal"``
    #: rebuilds the Group D block, the preprocessing fitted on it and the
    #: classifier, so it drives the like-for-like accuracy comparison reported
    #: in the manuscript rather than only the latency measurement.
    group_d_variant: str = "centered"

    def __post_init__(self) -> None:
        if self.group_d_variant not in {"centered", "causal"}:
            raise ValueError(
                f"group_d_variant must be 'centered' or 'causal', "
                f"got {self.group_d_variant!r}."
            )

    @property
    def group_d_causal(self) -> bool:
        """True when Group D uses the strictly causal trailing-window variant."""
        return self.group_d_variant == "causal"

    @property
    def xgb_overrides(self) -> dict:
        return {"n_estimators": int(self.xgb_n_estimators)}


@dataclass
class ExperimentContext:
    inputs: StudyInputs
    wells: WellsConfig
    labeler: FZILabeler
    labeled_core: pd.DataFrame
    partition: FullPartition
    development_labels: np.ndarray
    calibration_sample_ids: tuple[str, ...]
    extra_upstream_excluded_wells: tuple[str, ...] = ()

    @property
    def development(self) -> pd.DataFrame:
        return self.partition.development_df

    @property
    def blind(self) -> pd.DataFrame:
        return self.partition.blind_df

    @property
    def calibration_rows(self) -> pd.DataFrame:
        return self.development.iloc[self.partition.calibration.cp_calib_idx]

    @property
    def modeling_rows(self) -> pd.DataFrame:
        return self.development.iloc[self.partition.calibration.cv_pool_idx]


@dataclass
class PreparedOuterFold:
    fold: CVFold
    outer_train_rows: pd.DataFrame
    outer_val_rows: pd.DataFrame
    calibration_rows: pd.DataFrame
    X_outer_train: np.ndarray
    X_outer_val: np.ndarray
    X_calibration: np.ndarray
    y_outer_train: np.ndarray
    y_outer_val: np.ndarray
    y_calibration: np.ndarray
    gradient_local_idx: np.ndarray
    early_stop_local_idx: np.ndarray
    upstream_audits: tuple[UpstreamAudit, ...]


@dataclass
class OuterFoldRun:
    prepared: PreparedOuterFold
    train_result: FoldResult
    outer_prediction: PredictionResult
    calibration_prediction: PredictionResult
    metrics: ClassificationMetrics
    oof_frame: pd.DataFrame
    calibration_frame: pd.DataFrame


@dataclass
class PreparedFinalModel:
    modeling_rows: pd.DataFrame
    calibration_rows: pd.DataFrame
    blind_rows: pd.DataFrame
    X_modeling: np.ndarray
    X_calibration: np.ndarray
    X_blind: np.ndarray
    y_modeling: np.ndarray
    y_calibration: np.ndarray
    y_blind: np.ndarray
    gradient_local_idx: np.ndarray
    early_stop_local_idx: np.ndarray
    upstream_audits: tuple[UpstreamAudit, ...]


@dataclass
class FinalModelRun:
    prepared: PreparedFinalModel
    train_result: FoldResult
    blind_prediction: PredictionResult
    calibration_prediction: PredictionResult
    blind_metrics: ClassificationMetrics
    blind_frame: pd.DataFrame
    calibration_frame: pd.DataFrame


def build_context(
    data_dir: str | Path,
    *,
    cp_seed: int = 20260827,
    wells_cfg: WellsConfig | None = None,
    split_metadata_path: str | Path | None = None,
) -> ExperimentContext:
    """Load the three-file inputs, fit FZI labels on 890 rows, then partition."""
    wells = wells_cfg or load_wells_config()
    inputs = load_inputs(data_dir)
    core = match_core_to_continuous(inputs.core_samples, inputs.continuous_logs)
    labeler, labeled = fit_development_labels(core, wells.training_well_names)
    authoritative_ids = None
    if split_metadata_path is not None:
        meta = pd.read_csv(split_metadata_path)
        required = {"sample_id", "is_main_calibration"}
        missing = required - set(meta.columns)
        if missing:
            raise ValueError(f"split metadata missing columns: {sorted(missing)}")
        flag = meta["is_main_calibration"]
        if flag.dtype != bool:
            flag = flag.astype(str).str.strip().str.lower().map({"true": True, "false": False})
            if flag.isna().any():
                raise ValueError("is_main_calibration must contain boolean True/False values.")
        authoritative_ids = meta.loc[flag.to_numpy(), "sample_id"].astype(str).tolist()

    partition = build_full_partition(
        labeled,
        labeled["quality_class_id"].to_numpy(dtype=int),
        cp_seed=cp_seed,
        wells_cfg=wells,
        calibration_sample_ids=authoritative_ids,
    )
    dev_labels = partition.development_df["quality_class_id"].to_numpy(dtype=int)
    calibration_ids = tuple(
        partition.development_df.iloc[partition.calibration.cp_calib_idx]["sample_id"]
        .astype(str)
        .tolist()
    )
    return ExperimentContext(
        inputs=inputs,
        wells=wells,
        labeler=labeler,
        labeled_core=labeled,
        partition=partition,
        development_labels=dev_labels,
        calibration_sample_ids=calibration_ids,
    )


def context_with_calibration_sample_ids(
    base: ExperimentContext,
    calibration_sample_ids: tuple[str, ...] | list[str],
    *,
    extra_upstream_excluded_wells: tuple[str, ...] = (),
) -> ExperimentContext:
    """Clone an experiment context with an arbitrary Table-7 calibration set.

    FZI labels/boundaries and the seven-well data stay fixed.  Only the
    development rows assigned to calibration versus model fitting change.
    This helper intentionally does not create outer CV folds; Table 7 evaluates
    blind-well coverage after a full refit for each calibration design.
    """
    wanted = {str(x) for x in calibration_sample_ids}
    dev = base.development
    ids = dev["sample_id"].astype(str).to_numpy()
    missing = wanted - set(ids)
    if missing:
        raise ValueError(f"Calibration IDs not found in development data: {sorted(missing)[:5]}")
    calib = np.flatnonzero(np.isin(ids, list(wanted))).astype(np.int64)
    if len(calib) != len(wanted):
        raise ValueError("Calibration IDs are duplicated or did not select uniquely.")
    if len(calib) == 0 or len(calib) >= len(dev):
        raise ValueError("Calibration design must leave at least one modeling row.")
    pool = np.setdiff1d(np.arange(len(dev), dtype=np.int64), calib, assume_unique=True)
    partition = FullPartition(
        development_df=dev.copy(),
        blind_df=base.blind.copy(),
        calibration=CalibrationSplit(cv_pool_idx=pool, cp_calib_idx=np.sort(calib)),
        folds=(),
    )
    return ExperimentContext(
        inputs=base.inputs,
        wells=base.wells,
        labeler=base.labeler,
        labeled_core=base.labeled_core,
        partition=partition,
        development_labels=base.development_labels,
        calibration_sample_ids=tuple(dev.iloc[np.sort(calib)]["sample_id"].astype(str)),
        extra_upstream_excluded_wells=tuple(sorted(set(map(str, extra_upstream_excluded_wells)))),
    )


def _matched_group_d(rows: pd.DataFrame, d_cont: pd.DataFrame) -> pd.DataFrame:
    D = match_group_d_to_core(rows, d_cont)
    if D["sample_id"].astype(str).tolist() != rows["sample_id"].astype(str).tolist():
        raise AssertionError("Group-D/core row alignment changed during matching.")
    return D


def _runtime_train_cfg(options: RuntimeOptions) -> TrainConfig:
    return TrainConfig(max_epochs=int(options.max_epochs))


def _probability_frame(
    rows: pd.DataFrame,
    pred: PredictionResult,
    *,
    fold_id: int | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "sample_id": rows["sample_id"].astype(str).to_numpy(),
            "well_id": rows["well_id"].astype(str).to_numpy(),
            "quality_class_id": rows["quality_class_id"].to_numpy(dtype=int),
            "predicted_class_id": pred.predictions.astype(int),
            "prob_High": pred.probabilities[:, 0],
            "prob_Medium": pred.probabilities[:, 1],
            "prob_Low": pred.probabilities[:, 2],
            "alpha_A": pred.group_weights[:, 0],
            "alpha_B": pred.group_weights[:, 1],
            "alpha_C": pred.group_weights[:, 2],
            "alpha_D": pred.group_weights[:, 3],
        }
    )
    if fold_id is not None:
        out.insert(2, "fold", int(fold_id))
    return out


def prepare_outer_fold(
    context: ExperimentContext,
    fold: CVFold,
    *,
    options: RuntimeOptions | None = None,
) -> PreparedOuterFold:
    """Construct leakage-safe UFT matrices for one outer fold."""
    options = options or RuntimeOptions()
    dev = context.development
    outer_train = dev.iloc[fold.train_idx].copy()
    outer_val = dev.iloc[fold.val_idx].copy()
    calibration = context.calibration_rows.copy()

    # Training-row B/C: inner leave-one-well-out cross-fitting.
    train_proxy = crossfit_training_proxies(
        outer_train,
        context.inputs.lithology_corpus,
        calibration_sample_ids=context.calibration_sample_ids,
        always_exclude_wells=tuple(context.wells.blind_well_names) + tuple(context.extra_upstream_excluded_wells),
        outer_validation_well=fold.validation_well,
        well_order=context.wells.training_well_names,
        erf_n_estimators=options.erf_n_estimators,
        erf_max_depth=options.erf_max_depth,
        erf_random_state=options.erf_random_state,
        xgb_random_state=options.seed,
        xgb_overrides=options.xgb_overrides,
        max_lith_rows_per_class=options.max_lith_rows_per_class,
    )
    assert_crossfit_audits_no_target_well(train_proxy.audits)

    # Fold-specific upstream models for the held-out well and calibration rows.
    eval_proxy = fit_evaluation_upstream(
        outer_train,
        {"validation": outer_val, "calibration": calibration},
        context.inputs.lithology_corpus,
        calibration_sample_ids=context.calibration_sample_ids,
        always_exclude_wells=tuple(context.wells.blind_well_names) + tuple(context.extra_upstream_excluded_wells),
        evaluation_wells_to_exclude=[fold.validation_well],
        erf_n_estimators=options.erf_n_estimators,
        erf_max_depth=options.erf_max_depth,
        erf_random_state=options.erf_random_state,
        xgb_random_state=options.seed,
        xgb_overrides=options.xgb_overrides,
        max_lith_rows_per_class=options.max_lith_rows_per_class,
    )

    # A/D clipping + D context use only outer-training fitted limits.
    clipper = RawLogClipper().fit(outer_train)
    d_cont = build_group_d_continuous(
        context.inputs.continuous_logs, clipper=clipper, causal=options.group_d_causal
    )
    D_train = _matched_group_d(outer_train, d_cont)
    D_val = _matched_group_d(outer_val, d_cont)
    D_cal = _matched_group_d(calibration, d_cont)

    pre = UFTPreprocessor().fit_on(
        outer_train,
        train_proxy.group_b,
        train_proxy.group_c,
        D_train,
        raw_clipper=clipper,
    )
    X_train = pre.transform(
        outer_train, train_proxy.group_b, train_proxy.group_c, D_train
    )
    X_val = pre.transform(
        outer_val,
        eval_proxy["validation"].group_b,
        eval_proxy["validation"].group_c,
        D_val,
    )
    X_cal = pre.transform(
        calibration,
        eval_proxy["calibration"].group_b,
        eval_proxy["calibration"].group_c,
        D_cal,
    )

    y_train = outer_train["quality_class_id"].to_numpy(dtype=int)
    y_val = outer_val["quality_class_id"].to_numpy(dtype=int)
    y_cal = calibration["quality_class_id"].to_numpy(dtype=int)
    internal = make_outer_early_stopping_split(
        dev, context.development_labels, fold, seed=options.seed
    )

    # Explicit no-leakage role assertions.
    train_ids = set(outer_train["sample_id"].astype(str))
    val_ids = set(outer_val["sample_id"].astype(str))
    cal_ids = set(calibration["sample_id"].astype(str))
    blind_ids = set(context.blind["sample_id"].astype(str))
    if train_ids & val_ids or train_ids & cal_ids or train_ids & blind_ids:
        raise AssertionError("Outer training rows overlap an evaluation/calibration role.")

    audits = train_proxy.audits + eval_proxy["validation"].audits
    return PreparedOuterFold(
        fold=fold,
        outer_train_rows=outer_train,
        outer_val_rows=outer_val,
        calibration_rows=calibration,
        X_outer_train=X_train,
        X_outer_val=X_val,
        X_calibration=X_cal,
        y_outer_train=y_train,
        y_outer_val=y_val,
        y_calibration=y_cal,
        gradient_local_idx=internal.gradient_idx,
        early_stop_local_idx=internal.early_stop_idx,
        upstream_audits=audits,
    )


def train_outer_fold(
    context: ExperimentContext,
    fold: CVFold,
    *,
    options: RuntimeOptions | None = None,
    verbose: bool = False,
) -> OuterFoldRun:
    """Prepare, train, and evaluate one manuscript outer fold."""
    options = options or RuntimeOptions()
    p = prepare_outer_fold(context, fold, options=options)
    g = p.gradient_local_idx
    e = p.early_stop_local_idx
    result = train_one_fold(
        p.X_outer_train[g],
        p.y_outer_train[g],
        p.X_outer_train[e],
        p.y_outer_train[e],
        train_cfg=_runtime_train_cfg(options),
        seed=options.seed,
        class_weight_labels=p.y_outer_train,
        verbose=verbose,
    )
    outer_pred = predict_grqnet(result.model, p.X_outer_val)
    cal_pred = predict_grqnet(result.model, p.X_calibration)
    metrics = compute_metrics(p.y_outer_val, outer_pred.predictions)
    return OuterFoldRun(
        prepared=p,
        train_result=result,
        outer_prediction=outer_pred,
        calibration_prediction=cal_pred,
        metrics=metrics,
        oof_frame=_probability_frame(
            p.outer_val_rows, outer_pred, fold_id=p.fold.fold_id
        ),
        calibration_frame=_probability_frame(
            p.calibration_rows, cal_pred, fold_id=p.fold.fold_id
        ),
    )


def prepare_final_model(
    context: ExperimentContext,
    *,
    options: RuntimeOptions | None = None,
) -> PreparedFinalModel:
    """Construct final-model UFT plus calibration/blind matrices for the active design."""
    options = options or RuntimeOptions()
    modeling = context.modeling_rows.copy()
    calibration = context.calibration_rows.copy()
    blind = context.blind.copy()

    train_proxy = crossfit_training_proxies(
        modeling,
        context.inputs.lithology_corpus,
        calibration_sample_ids=context.calibration_sample_ids,
        always_exclude_wells=tuple(context.wells.blind_well_names) + tuple(context.extra_upstream_excluded_wells),
        outer_validation_well=None,
        well_order=context.wells.training_well_names,
        erf_n_estimators=options.erf_n_estimators,
        erf_max_depth=options.erf_max_depth,
        erf_random_state=options.erf_random_state,
        xgb_random_state=options.seed,
        xgb_overrides=options.xgb_overrides,
        max_lith_rows_per_class=options.max_lith_rows_per_class,
    )
    assert_crossfit_audits_no_target_well(train_proxy.audits)

    deployment_proxy = fit_evaluation_upstream(
        modeling,
        {"calibration": calibration, "blind": blind},
        context.inputs.lithology_corpus,
        calibration_sample_ids=context.calibration_sample_ids,
        always_exclude_wells=tuple(context.wells.blind_well_names) + tuple(context.extra_upstream_excluded_wells),
        evaluation_wells_to_exclude=(),
        erf_n_estimators=options.erf_n_estimators,
        erf_max_depth=options.erf_max_depth,
        erf_random_state=options.erf_random_state,
        xgb_random_state=options.seed,
        xgb_overrides=options.xgb_overrides,
        max_lith_rows_per_class=options.max_lith_rows_per_class,
    )

    clipper = RawLogClipper().fit(modeling)
    d_cont = build_group_d_continuous(
        context.inputs.continuous_logs, clipper=clipper, causal=options.group_d_causal
    )
    D_train = _matched_group_d(modeling, d_cont)
    D_cal = _matched_group_d(calibration, d_cont)
    D_blind = _matched_group_d(blind, d_cont)

    pre = UFTPreprocessor().fit_on(
        modeling,
        train_proxy.group_b,
        train_proxy.group_c,
        D_train,
        raw_clipper=clipper,
    )
    X_train = pre.transform(
        modeling, train_proxy.group_b, train_proxy.group_c, D_train
    )
    X_cal = pre.transform(
        calibration,
        deployment_proxy["calibration"].group_b,
        deployment_proxy["calibration"].group_c,
        D_cal,
    )
    X_blind = pre.transform(
        blind,
        deployment_proxy["blind"].group_b,
        deployment_proxy["blind"].group_c,
        D_blind,
    )

    y_model = modeling["quality_class_id"].to_numpy(dtype=int)
    y_cal = calibration["quality_class_id"].to_numpy(dtype=int)
    y_blind = blind["quality_class_id"].to_numpy(dtype=int)
    if len(y_model) == 712 and len(calibration) == 178:
        internal = make_final_early_stopping_split(y_model, seed=options.seed)
    else:
        # Table-7 alternative calibration designs retain the same 15%
        # class-stratified checkpoint-selection protocol, with ceil rounding.
        internal = make_early_stopping_split(y_model, fraction=0.15, seed=options.seed)

    model_ids = set(modeling["sample_id"].astype(str))
    cal_ids = set(calibration["sample_id"].astype(str))
    blind_ids = set(blind["sample_id"].astype(str))
    if model_ids & cal_ids or model_ids & blind_ids or cal_ids & blind_ids:
        raise AssertionError("Final model/calibration/blind sample roles overlap.")

    audits = train_proxy.audits + deployment_proxy["blind"].audits
    return PreparedFinalModel(
        modeling_rows=modeling,
        calibration_rows=calibration,
        blind_rows=blind,
        X_modeling=X_train,
        X_calibration=X_cal,
        X_blind=X_blind,
        y_modeling=y_model,
        y_calibration=y_cal,
        y_blind=y_blind,
        gradient_local_idx=internal.gradient_idx,
        early_stop_local_idx=internal.early_stop_idx,
        upstream_audits=audits,
    )


def train_final_model(
    context: ExperimentContext,
    *,
    options: RuntimeOptions | None = None,
    verbose: bool = False,
) -> FinalModelRun:
    """Train the active final neural model and evaluate the blind wells."""
    options = options or RuntimeOptions()
    p = prepare_final_model(context, options=options)
    g = p.gradient_local_idx
    e = p.early_stop_local_idx
    result = train_one_fold(
        p.X_modeling[g],
        p.y_modeling[g],
        p.X_modeling[e],
        p.y_modeling[e],
        train_cfg=_runtime_train_cfg(options),
        seed=options.seed,
        class_weight_labels=p.y_modeling,
        verbose=verbose,
    )
    blind_pred = predict_grqnet(result.model, p.X_blind)
    cal_pred = predict_grqnet(result.model, p.X_calibration)
    metrics = compute_metrics(p.y_blind, blind_pred.predictions)
    return FinalModelRun(
        prepared=p,
        train_result=result,
        blind_prediction=blind_pred,
        calibration_prediction=cal_pred,
        blind_metrics=metrics,
        blind_frame=_probability_frame(p.blind_rows, blind_pred),
        calibration_frame=_probability_frame(p.calibration_rows, cal_pred),
    )


def run_main_experiment(
    data_dir: str | Path,
    *,
    options: RuntimeOptions | None = None,
    include_final_model: bool = True,
    verbose: bool = False,
    split_metadata_path: str | Path | None = None,
) -> tuple[ExperimentContext, list[OuterFoldRun], FinalModelRun | None]:
    """Run five outer folds and optionally the final blind-well model."""
    options = options or RuntimeOptions()
    context = build_context(
        data_dir, cp_seed=options.seed, split_metadata_path=split_metadata_path
    )
    folds = [
        train_outer_fold(context, fold, options=options, verbose=verbose)
        for fold in context.partition.folds
    ]
    final = (
        train_final_model(context, options=options, verbose=verbose)
        if include_final_model
        else None
    )
    return context, folds, final


# Backward-compatible aliases for stale external imports.
def train_blind_model(context: ExperimentContext, **kwargs) -> FinalModelRun:
    return train_final_model(context, **kwargs)


def predict_blind(final_run: FinalModelRun) -> pd.DataFrame:
    return final_run.blind_frame.copy()


def evaluate_blind(final_run: FinalModelRun) -> ClassificationMetrics:
    return final_run.blind_metrics
