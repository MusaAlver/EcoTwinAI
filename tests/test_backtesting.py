
import numpy as np
import pandas as pd
import pytest

from src.backtesting import WalkForwardBacktester
from src.training_data import TimeSeriesDatasetBuilder


def make_frame(rows: int = 1200) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="15min",
            ),
            "total_power": (
                40.0
                + 0.02 * index
                + 2.0 * np.sin(
                    2 * np.pi * index / 96
                )
            ),
        }
    )


def make_backtester(
    **kwargs,
) -> WalkForwardBacktester:
    defaults = {
        "min_train_rows": 500,
        "validation_rows": 100,
        "step_rows": 100,
        "max_folds": 4,
    }

    defaults.update(kwargs)

    return WalkForwardBacktester(
        **defaults
    )


def test_walk_forward_builds_expanding_folds():
    frame = make_frame()

    backtester = make_backtester()

    folds = backtester.build_folds(
        frame,
        feature_columns=[
            "total_power",
        ],
    )

    assert len(folds) == 4

    train_rows = [
        fold.train_rows
        for fold, *_ in folds
    ]

    assert train_rows == [
        500,
        600,
        700,
        800,
    ]


def test_validation_windows_do_not_overlap():
    frame = make_frame()

    backtester = make_backtester()

    folds = backtester.build_folds(
        frame,
        feature_columns=[
            "total_power",
        ],
    )

    previous_end = None

    for fold, *_ in folds:
        if previous_end is not None:
            assert (
                fold.validation_start
                > previous_end
            )

        previous_end = (
            fold.validation_end
        )


def test_training_outcomes_never_enter_validation_period():
    frame = make_frame()

    backtester = make_backtester()

    folds = backtester.build_folds(
        frame,
        feature_columns=[
            "total_power",
        ],
    )

    for (
        fold,
        train_dataset,
        validation_dataset,
        _,
    ) in folds:
        train_outcomes = pd.to_datetime(
            train_dataset.outcome_time
        )

        validation_outcomes = pd.to_datetime(
            validation_dataset.outcome_time
        )

        assert (
            train_outcomes.max()
            < fold.validation_start
        )

        assert (
            validation_outcomes.min()
            >= fold.validation_start
        )

        assert (
            validation_outcomes.max()
            <= fold.validation_end
        )


def test_validation_can_use_known_training_history():
    frame = make_frame()

    builder = TimeSeriesDatasetBuilder(
        lookback_steps=16,
        horizon_minutes=30,
        sampling_minutes=15,
    )

    backtester = make_backtester(
        dataset_builder=builder
    )

    folds = backtester.build_folds(
        frame,
        feature_columns=[
            "total_power",
        ],
    )

    (
        fold,
        _,
        validation_dataset,
        _,
    ) = folds[0]

    first_forecast = pd.Timestamp(
        validation_dataset.forecast_time[0]
    )

    first_outcome = pd.Timestamp(
        validation_dataset.outcome_time[0]
    )

    assert (
        first_forecast
        < fold.validation_start
    )

    assert (
        first_outcome
        == fold.validation_start
    )


def test_perfect_candidate_is_evaluated_on_every_fold():
    frame = make_frame()

    backtester = make_backtester()

    seen_folds = []

    def perfect_fit_predict(
        train_dataset,
        validation_dataset,
        context,
    ):
        seen_folds.append(
            context["fold"]
        )

        assert (
            pd.Timestamp(
                train_dataset.outcome_time[-1]
            )
            < context["validation_start"]
        )

        return validation_dataset.y.copy()

    result = backtester.run(
        frame,
        feature_columns=[
            "total_power",
        ],
        fit_predict=perfect_fit_predict,
    )

    assert seen_folds == [
        1,
        2,
        3,
        4,
    ]

    assert (
        result["summary"]["fold_count"]
        == 4
    )

    assert (
        result["summary"]
        ["aggregate_candidate"]
        ["mae"]
        == pytest.approx(0.0)
    )

    assert (
        result["summary"]
        ["validation_windows_overlap"]
        is False
    )


def test_wrong_prediction_count_is_rejected():
    frame = make_frame()

    backtester = make_backtester(
        max_folds=1
    )

    def bad_fit_predict(
        train_dataset,
        validation_dataset,
        context,
    ):
        return np.zeros(
            len(validation_dataset) + 1
        )

    with pytest.raises(
        ValueError,
        match="unexpected number",
    ):
        backtester.run(
            frame,
            feature_columns=[
                "total_power",
            ],
            fit_predict=bad_fit_predict,
        )


def test_overlapping_validation_configuration_is_rejected():
    with pytest.raises(
        ValueError,
        match="prevent overlapping",
    ):
        WalkForwardBacktester(
            min_train_rows=500,
            validation_rows=100,
            step_rows=50,
        )


def test_dataset_too_short_for_backtest_is_rejected():
    frame = make_frame(
        rows=550
    )

    backtester = make_backtester(
        min_train_rows=500,
        validation_rows=100,
    )

    with pytest.raises(
        ValueError,
        match="too short",
    ):
        backtester.build_folds(
            frame,
            feature_columns=[
                "total_power",
            ],
        )
