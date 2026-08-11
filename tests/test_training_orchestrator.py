
import numpy as np
import pandas as pd
import pytest

from src.model_registry import ModelRegistry
from src.training_data import SequenceDataset, TimeSeriesDatasetBuilder
from src.training_orchestrator import TrainingOrchestrator


def make_frame(
    rows: int = 3000,
):
    timestamp = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="15min",
    )

    index = np.arange(
        rows,
        dtype=float,
    )

    power = (
        50.0
        + 5.0 * np.sin(
            index * 2 * np.pi / 96
        )
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "total_power": power,
            "time_feature": index,
        }
    )


def make_sequence_dataset(
    frame: pd.DataFrame,
    outcome_indices: list[int],
) -> SequenceDataset:
    forecast_indices = [
        index - 2
        for index in outcome_indices
    ]

    X = np.zeros(
        (
            len(outcome_indices),
            16,
            1,
        ),
        dtype=np.float32,
    )

    y = (
        frame.iloc[outcome_indices]
        ["total_power"]
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )

    forecast_time = (
        frame.iloc[forecast_indices]
        ["timestamp"]
        .to_numpy()
    )

    outcome_time = (
        frame.iloc[outcome_indices]
        ["timestamp"]
        .to_numpy()
    )

    return SequenceDataset(
        X=X,
        y=y,
        forecast_time=forecast_time,
        outcome_time=outcome_time,
    )


def test_prediction_metrics_are_correct():
    actual = np.array(
        [10.0, 20.0, 30.0]
    )

    prediction = np.array(
        [12.0, 18.0, 34.0]
    )

    result = (
        TrainingOrchestrator
        .evaluate_predictions(
            actual,
            prediction,
        )
    )

    assert result.samples == 3
    assert result.mae == pytest.approx(
        8 / 3
    )

    assert result.rmse == pytest.approx(
        np.sqrt(
            24 / 3
        )
    )

    assert result.within_5kw == 1.0
    assert result.within_10kw == 1.0


def test_predict_rejects_wrong_sample_count(
    tmp_path,
):
    orchestrator = TrainingOrchestrator(
        registry=ModelRegistry(
            tmp_path / "registry"
        )
    )

    frame = make_frame(
        rows=100
    )

    dataset = make_sequence_dataset(
        frame,
        outcome_indices=[
            30,
            31,
            32,
        ],
    )

    def bad_predictor(X):
        return np.zeros(
            (
                len(X) + 1,
                1,
            )
        )

    with pytest.raises(
        ValueError,
        match="unexpected number of samples",
    ):
        orchestrator.predict(
            bad_predictor,
            dataset,
        )


def test_aligned_baselines_use_correct_reference_times(
    tmp_path,
):
    frame = make_frame(
        rows=900
    )

    dataset = make_sequence_dataset(
        frame,
        outcome_indices=[
            700,
            701,
            702,
            703,
        ],
    )

    orchestrator = TrainingOrchestrator(
        registry=ModelRegistry(
            tmp_path / "registry"
        )
    )

    results = orchestrator.aligned_baselines(
        full_frame=frame,
        dataset=dataset,
    )

    assert "persistence" in results
    assert "daily_seasonal" in results
    assert "weekly_seasonal" in results

    assert (
        results["persistence"]["samples"]
        == len(dataset)
    )

    assert (
        results["daily_seasonal"]["samples"]
        == len(dataset)
    )

    assert (
        results["weekly_seasonal"]["samples"]
        == len(dataset)
    )


def test_partial_history_does_not_allow_unfair_sample_comparison(
    tmp_path,
):
    frame = make_frame(
        rows=250
    )

    dataset = make_sequence_dataset(
        frame,
        outcome_indices=[
            94,
            95,
            96,
            97,
            98,
            99,
            100,
        ],
    )

    orchestrator = TrainingOrchestrator(
        registry=ModelRegistry(
            tmp_path / "registry"
        )
    )

    candidate_prediction = (
        dataset.y.copy()
    )

    result = orchestrator.evaluate_candidate(
        candidate_prediction=candidate_prediction,
        dataset=dataset,
        full_frame=frame,
    )

    candidate_samples = (
        result["candidate"]["samples"]
    )

    for baseline in (
        result["baselines"].values()
    ):
        assert (
            baseline["samples"]
            == candidate_samples
        )


def test_perfect_candidate_is_promoted(
    tmp_path,
):
    frame = make_frame(
        rows=900
    )

    dataset = make_sequence_dataset(
        frame,
        outcome_indices=list(
            range(
                700,
                720,
            )
        ),
    )

    orchestrator = TrainingOrchestrator(
        registry=ModelRegistry(
            tmp_path / "registry"
        ),
        minimum_improvement=0.01,
    )

    result = orchestrator.evaluate_candidate(
        candidate_prediction=dataset.y.copy(),
        dataset=dataset,
        full_frame=frame,
    )

    assert (
        result["promotion_decision"]
        ["accepted"]
        is True
    )

    assert (
        result["promotion_decision"]
        ["decision"]
        == "PROMOTE"
    )


def test_bad_candidate_is_rejected(
    tmp_path,
):
    frame = make_frame(
        rows=900
    )

    dataset = make_sequence_dataset(
        frame,
        outcome_indices=list(
            range(
                700,
                720,
            )
        ),
    )

    prediction = (
        dataset.y
        + 50.0
    )

    orchestrator = TrainingOrchestrator(
        registry=ModelRegistry(
            tmp_path / "registry"
        ),
        minimum_improvement=0.01,
    )

    result = orchestrator.evaluate_candidate(
        candidate_prediction=prediction,
        dataset=dataset,
        full_frame=frame,
    )

    assert (
        result["promotion_decision"]
        ["accepted"]
        is False
    )

    assert (
        result["promotion_decision"]
        ["decision"]
        == "REJECT"
    )


def test_validation_evaluation_does_not_use_test_split(
    tmp_path,
):
    builder = TimeSeriesDatasetBuilder(
        lookback_steps=16,
        horizon_minutes=30,
        sampling_minutes=15,
    )

    orchestrator = TrainingOrchestrator(
        dataset_builder=builder,
        registry=ModelRegistry(
            tmp_path / "registry"
        ),
    )

    frame = make_frame()

    prepared = (
        orchestrator
        .prepare_training_data(
            frame,
            feature_columns=[
                "total_power",
                "time_feature",
            ],
        )
    )

    def perfect_predictor(X):
        validation = (
            prepared["datasets"]
            ["validation"]
        )

        return validation.y.copy()

    original = (
        orchestrator
        .evaluate_validation(
            prepared=prepared,
            predictor=perfect_predictor,
        )
    )

    prepared["splits"]["test"][
        "total_power"
    ] = 999999.0

    changed = (
        orchestrator
        .evaluate_validation(
            prepared=prepared,
            predictor=perfect_predictor,
        )
    )

    assert (
        original["candidate"]
        == changed["candidate"]
    )

    assert (
        original["promotion_decision"]
        == changed["promotion_decision"]
    )


def test_test_evaluation_has_no_promotion_decision(
    tmp_path,
):
    builder = TimeSeriesDatasetBuilder()

    orchestrator = TrainingOrchestrator(
        dataset_builder=builder,
        registry=ModelRegistry(
            tmp_path / "registry"
        ),
    )

    prepared = (
        orchestrator
        .prepare_training_data(
            make_frame(),
            feature_columns=[
                "total_power",
                "time_feature",
            ],
        )
    )

    test_dataset = (
        prepared["datasets"]["test"]
    )

    def perfect_predictor(X):
        return test_dataset.y.copy()

    result = orchestrator.evaluate_test(
        prepared=prepared,
        predictor=perfect_predictor,
    )

    assert result["mae"] == pytest.approx(
        0.0
    )

    assert "promotion_decision" not in result
    assert "baseline_champion" not in result
