"""Round-3 leakage barriers for outer/inner upstream fitting and checkpointing."""

from __future__ import annotations

import pytest

from scripts.generate_synthetic_data import write_synthetic_inputs
from src.pipeline import RuntimeOptions, build_context, prepare_final_model, prepare_outer_fold
from src.training.upstream import filter_lithology_corpus


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    d = tmp_path_factory.mktemp("round3_leakage_inputs")
    write_synthetic_inputs(d, 20260827)
    return build_context(d)


@pytest.fixture(scope="module")
def smoke_options():
    return RuntimeOptions(
        erf_n_estimators=1,
        erf_max_depth=4,
        xgb_n_estimators=2,
        max_lith_rows_per_class=4,
        max_epochs=1,
    )


def test_erf_filter_excludes_blind_calibration_and_outer_validation(context):
    outer_val = "WF1"
    filtered = filter_lithology_corpus(
        context.inputs.lithology_corpus,
        exclude_wells=[*context.wells.blind_well_names, outer_val],
        exclude_source_sample_ids=context.calibration_sample_ids,
    )
    assert not set(context.wells.blind_well_names) & set(filtered["well_id"])
    assert outer_val not in set(filtered["well_id"])
    source_ids = set(filtered["source_sample_id"].astype(str))
    assert source_ids.isdisjoint(context.calibration_sample_ids)


def test_outer_fold_upstream_and_checkpoint_roles_are_disjoint(context, smoke_options):
    fold = context.partition.folds[0]
    p = prepare_outer_fold(context, fold, options=smoke_options)

    # Every inner target well is absent from both Group-B and Group-C fit wells.
    for audit in p.upstream_audits:
        if audit.purpose.startswith("inner_crossfit:"):
            target = set(audit.target_wells)
            assert target.isdisjoint(audit.core_fit_wells)
            assert target.isdisjoint(audit.lithology_fit_wells)

    eval_audit = [a for a in p.upstream_audits if a.purpose == "evaluation_upstream"][0]
    assert fold.validation_well not in set(eval_audit.core_fit_wells)
    assert fold.validation_well not in set(eval_audit.lithology_fit_wells)
    assert set(context.wells.blind_well_names).isdisjoint(eval_audit.lithology_fit_wells)

    train_ids = set(p.outer_train_rows["sample_id"].astype(str))
    val_ids = set(p.outer_val_rows["sample_id"].astype(str))
    cal_ids = set(p.calibration_rows["sample_id"].astype(str))
    blind_ids = set(context.blind["sample_id"].astype(str))
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(cal_ids)
    assert train_ids.isdisjoint(blind_ids)

    gradient_ids = set(
        p.outer_train_rows.iloc[p.gradient_local_idx]["sample_id"].astype(str)
    )
    early_ids = set(
        p.outer_train_rows.iloc[p.early_stop_local_idx]["sample_id"].astype(str)
    )
    assert gradient_ids.isdisjoint(early_ids)
    assert gradient_ids.isdisjoint(val_ids | cal_ids | blind_ids)
    assert early_ids.isdisjoint(val_ids | cal_ids | blind_ids)


def test_final_model_uses_only_712_for_fit_and_605_107_neural_split(context, smoke_options):
    p = prepare_final_model(context, options=smoke_options)
    assert len(p.modeling_rows) == 712
    assert len(p.calibration_rows) == 178
    assert len(p.blind_rows) == 369
    assert len(p.gradient_local_idx) == 605
    assert len(p.early_stop_local_idx) == 107

    model_ids = set(p.modeling_rows["sample_id"].astype(str))
    cal_ids = set(p.calibration_rows["sample_id"].astype(str))
    blind_ids = set(p.blind_rows["sample_id"].astype(str))
    assert model_ids.isdisjoint(cal_ids)
    assert model_ids.isdisjoint(blind_ids)
    assert cal_ids.isdisjoint(blind_ids)

    deployment = [a for a in p.upstream_audits if a.purpose == "evaluation_upstream"][0]
    assert set(context.wells.blind_well_names).isdisjoint(deployment.core_fit_wells)
    assert set(context.wells.blind_well_names).isdisjoint(deployment.lithology_fit_wells)
