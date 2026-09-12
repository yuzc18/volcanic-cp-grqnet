from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def _contains_value(spec, value):
    kind = spec["type"]
    if kind == "categorical":
        return value in spec["choices"]
    if kind in {"int", "float"}:
        low, high = spec["low"], spec["high"]
        if not (low <= value <= high):
            return False
        step = spec.get("step")
        if step is None:
            return True
        # tolerate decimal representation for stepped floating domains
        q = (value - low) / step
        return abs(q - round(q)) < 1e-8
    raise AssertionError(f"unsupported type {kind}")


def test_search_protocol_matches_manuscript_counts_and_selection_rule():
    cfg = _load_yaml("configs/search_spaces.yaml")
    sp = cfg["search_protocol"]

    for name in ("lightgbm", "xgboost", "catboost"):
        assert sp[name]["engine"] == "optuna"
        assert sp[name]["trials_per_outer_fold"] == 50
        assert sp[name]["available"] is True
        assert sp[name]["role"] == "manuscript_release_search_space"
        assert sp[name]["space"]

    assert sp["multinomial_logistic_regression"]["engine"] == "grid"
    assert sp["multinomial_logistic_regression"]["n_configurations"] == 4
    assert len(sp["multinomial_logistic_regression"]["space"]) == 4

    assert sp["random_forest"]["engine"] == "grid"
    assert sp["random_forest"]["n_configurations"] == 18
    assert len(sp["random_forest"]["space"]) == 18

    assert cfg["selection_metric"] == "macro_f1"
    assert cfg["selection_split"] == "internal_15_percent_stratified"
    assert cfg["final_non_neural_protocol"] == "tune_on_712_then_refit_on_all_712"


def test_table3a_final_configs_are_contained_in_authoritative_spaces():
    search = _load_yaml("configs/search_spaces.yaml")["search_protocol"]
    baseline = _load_yaml("configs/baselines.yaml")["non_neural_baselines"]

    for model_name, bcfg in baseline.items():
        final = bcfg["table3a_final"]
        entry = search[model_name]
        if entry["engine"] == "grid":
            assert final in entry["space"], f"{model_name} Table 3(a) final config is absent from grid"
            continue

        by_name = {spec["name"]: spec for spec in entry["space"]}
        assert set(final) == set(by_name), f"{model_name} search dimensions differ from Table 3(a) final fields"
        for key, value in final.items():
            assert _contains_value(by_name[key], value), f"{model_name}.{key}={value} is outside search space"
