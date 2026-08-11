
import numpy as np
import pandas as pd
import pytest

from src.baselines import BaselineEvaluator


def test_persistence_metrics_are_correct():
    power = pd.Series(
        [10.0, 20.0, 30.0, 40.0, 50.0]
    )

    evaluator = BaselineEvaluator(
        sampling_minutes=15,
        horizon_minutes=30,
    )

    result = evaluator.evaluate_persistence(power)

    assert result.samples == 3
    assert result.mae == pytest.approx(20.0)
    assert result.rmse == pytest.approx(20.0)
    assert result.within_5kw == 0.0
    assert result.within_10kw == 0.0


def test_daily_seasonal_repeated_pattern_is_perfect():
    daily_pattern = np.arange(
        96,
        dtype=float,
    )

    power = pd.Series(
        np.tile(
            daily_pattern,
            3,
        )
    )

    evaluator = BaselineEvaluator(
        sampling_minutes=15,
        horizon_minutes=30,
    )

    result = evaluator.evaluate_seasonal(
        power,
        period_steps=96,
        name="daily_seasonal",
    )

    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.within_5kw == 1.0
    assert result.within_10kw == 1.0


def test_weekly_seasonal_repeated_pattern_is_perfect():
    weekly_pattern = np.arange(
        672,
        dtype=float,
    )

    power = pd.Series(
        np.tile(
            weekly_pattern,
            2,
        )
    )

    evaluator = BaselineEvaluator()

    result = evaluator.evaluate_seasonal(
        power,
        period_steps=672,
        name="weekly_seasonal",
    )

    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)


def test_evaluate_all_returns_available_baselines():
    rows = 800

    df = pd.DataFrame(
        {
            "total_power": (
                50
                + np.sin(
                    np.arange(rows) / 10
                )
            )
        }
    )

    evaluator = BaselineEvaluator()

    results = evaluator.evaluate_all(df)

    assert "persistence" in results
    assert "daily_seasonal" in results
    assert "weekly_seasonal" in results


def test_select_best_uses_lowest_mae():
    results = {
        "persistence": {
            "mae": 3.0,
            "rmse": 4.0,
        },
        "daily_seasonal": {
            "mae": 2.0,
            "rmse": 3.5,
        },
        "weekly_seasonal": {
            "mae": 2.5,
            "rmse": 3.0,
        },
    }

    best = BaselineEvaluator.select_best(
        results,
        metric="mae",
    )

    assert best["name"] == "daily_seasonal"
    assert best["value"] == 2.0


def test_candidate_is_promoted_when_it_beats_baseline():
    baseline = {
        "name": "persistence",
        "metric": "mae",
        "value": 4.0,
    }

    result = BaselineEvaluator.compare_candidate(
        {
            "mae": 3.0,
        },
        baseline,
        minimum_improvement=0.05,
    )

    assert result["accepted"] is True
    assert result["decision"] == "PROMOTE"
    assert result["relative_improvement"] == pytest.approx(
        0.25
    )


def test_candidate_is_rejected_when_gain_is_too_small():
    baseline = {
        "name": "persistence",
        "metric": "mae",
        "value": 4.0,
    }

    result = BaselineEvaluator.compare_candidate(
        {
            "mae": 3.98,
        },
        baseline,
        minimum_improvement=0.01,
    )

    assert result["accepted"] is False
    assert result["decision"] == "REJECT"


def test_invalid_horizon_is_rejected():
    with pytest.raises(
        ValueError,
        match="divisible",
    ):
        BaselineEvaluator(
            sampling_minutes=15,
            horizon_minutes=20,
        )
