
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .forecast import EcoTwinForecaster
from .uncertainty import AdaptiveConformalThreshold
from .anomaly import EcoTwinAnomalyDetector
from .root_cause import (
    EcoTwinRootCauseEngine,
    COMPONENTS,
)
from .recommendation import EcoTwinRecommendationEngine
from .incidents import EcoTwinIncidentEngine


class EcoTwinEngine:
    """
    Complete EcoTwin AI runtime engine.

    Pipeline
    --------
    Forecast
        ↓
    Adaptive uncertainty
        ↓
    Observed outcome
        ↓
    Anomaly detection
        ↓
    Root-cause attribution
        ↓
    Recommendation
        ↓
    Incident aggregation
    """

    def __init__(
        self,
        project_root=None,
    ):

        if project_root is None:

            project_root = (
                Path(__file__)
                .resolve()
                .parents[1]
            )

        self.project_root = Path(
            project_root
        )

        self.model_dir = (
            self.project_root
            / "models"
        )

        # ====================================================
        # FORECAST
        # ====================================================

        self.forecaster = (
            EcoTwinForecaster(
                self.project_root
            )
        )

        self.features = list(
            self.forecaster.features
        )

        self.feature_index = {
            name: i
            for i, name in enumerate(
                self.features
            )
        }

        self.component_indices = [
            self.feature_index[name]
            for name in COMPONENTS
        ]

        # ====================================================
        # CONFIG
        # ====================================================

        anomaly_config = self._load_json(
            "ecotwin_anomaly_config.json",
            {
                "coverage": 0.96,
                "rolling_window_days": 30,
                "forecast_horizon_minutes": 30,
            },
        )

        root_config = self._load_json(
            "ecotwin_root_cause_config.json",
            {
                "window_days": 28,
                "minimum_scale_kw": 0.25,
                "local_scale_floor_ratio": 0.20,
            },
        )

        incident_config = self._load_json(
            "ecotwin_incident_config.json",
            {
                "max_alarm_gap_minutes": 30,
                "observation_interval_minutes": 15,
            },
        )

        # ====================================================
        # ENGINES
        # ====================================================

        self.uncertainty = (
            AdaptiveConformalThreshold(
                coverage=float(
                    anomaly_config.get(
                        "coverage",
                        0.96,
                    )
                ),

                window_days=int(
                    anomaly_config.get(
                        "rolling_window_days",
                        30,
                    )
                ),

                horizon_minutes=int(
                    anomaly_config.get(
                        "forecast_horizon_minutes",
                        30,
                    )
                ),
            )
        )

        self.anomaly_detector = (
            EcoTwinAnomalyDetector()
        )

        self.root_cause = (
            EcoTwinRootCauseEngine(
                window_days=int(
                    root_config.get(
                        "window_days",
                        28,
                    )
                ),

                minimum_scale=float(
                    root_config.get(
                        "minimum_scale_kw",
                        0.25,
                    )
                ),

                local_scale_floor_ratio=float(
                    root_config.get(
                        "local_scale_floor_ratio",
                        0.20,
                    )
                ),
            )
        )

        self.recommendation = (
            EcoTwinRecommendationEngine(
                self.features
            )
        )

        self.incidents = (
            EcoTwinIncidentEngine(
                max_gap_minutes=int(
                    incident_config.get(
                        "max_alarm_gap_minutes",
                        30,
                    )
                ),

                observation_minutes=int(
                    incident_config.get(
                        "observation_interval_minutes",
                        15,
                    )
                ),
            )
        )

        # ====================================================
        # RUNTIME STATE
        # ====================================================

        self.pending_forecasts = {}

        self.alarm_rows = []

        self.initialized = False


    # ========================================================
    # JSON HELPER
    # ========================================================

    def _load_json(
        self,
        filename,
        default,
    ):

        path = (
            self.model_dir
            / filename
        )

        if not path.exists():
            return default

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)


    # ========================================================
    # INITIAL CALIBRATION
    # ========================================================

    def initialize(
        self,
        calibration_forecast_dates,
        calibration_absolute_errors,
        component_dates,
        component_values,
        production_start,
    ):

        production_start = pd.Timestamp(
            production_start
        )

        # Adaptive forecast-error history
        initial_threshold = (
            self.uncertainty.initialize(
                forecast_dates=
                    calibration_forecast_dates,

                absolute_errors=
                    calibration_absolute_errors,

                production_start=
                    production_start,
            )
        )

        # Root-cause subsystem history
        self.root_cause.initialize(
            dates=
                component_dates,

            component_values=
                component_values,

            production_start=
                production_start,
        )

        self.pending_forecasts.clear()
        self.alarm_rows.clear()

        self.initialized = True

        return {
            "initialized": True,

            "production_start":
                production_start.isoformat(),

            "initial_threshold_kw":
                float(
                    initial_threshold
                ),

            "feature_count":
                len(
                    self.features
                ),

            "component_count":
                len(
                    COMPONENTS
                ),
        }


    # ========================================================
    # CREATE 30-MIN FORECAST
    # ========================================================

    def create_forecast(
        self,
        forecast_time,
        raw_window,
        persistence_kw=None,
    ):

        if not self.initialized:

            raise RuntimeError(
                "EcoTwinEngine must be "
                "initialized first."
            )

        forecast_time = pd.Timestamp(
            forecast_time
        )

        # Flush outcomes that are already known,
        # prune history and calculate the current limit.
        threshold_kw = (
            self.uncertainty
            .current_threshold(
                forecast_time
            )
        )

        forecast_result = (
            self.forecaster
            .predict_30m(
                raw_window,
                persistence_kw=
                    persistence_kw,
            )
        )

        forecast_kw = float(
            forecast_result[
                "forecast_kw"
            ]
        )

        outcome_time = (
            forecast_time
            +
            pd.Timedelta(
                minutes=30
            )
        )

        forecast_id = (
            forecast_time.isoformat()
        )

        self.pending_forecasts[
            forecast_id
        ] = {
            "forecast_time":
                forecast_time,

            "outcome_time":
                outcome_time,

            "forecast_kw":
                forecast_kw,

            "threshold_kw":
                float(
                    threshold_kw
                ),

            "persistence_kw":
                float(
                    forecast_result[
                        "persistence_kw"
                    ]
                ),
        }

        return {
            **forecast_result,

            "forecast_id":
                forecast_id,

            "forecast_time":
                forecast_time.isoformat(),

            "outcome_time":
                outcome_time.isoformat(),

            "adaptive_threshold_kw":
                float(
                    threshold_kw
                ),

            "lower_bound_kw":
                float(
                    forecast_kw
                    -
                    threshold_kw
                ),

            "upper_bound_kw":
                float(
                    forecast_kw
                    +
                    threshold_kw
                ),
        }


    # ========================================================
    # PROCESS REAL OUTCOME
    # ========================================================

    def process_outcome(
        self,
        forecast_id,
        observed_state,
    ):

        if not self.initialized:

            raise RuntimeError(
                "EcoTwinEngine must be "
                "initialized first."
            )

        forecast_id = (
            pd.Timestamp(
                forecast_id
            )
            .isoformat()
        )

        if (
            forecast_id
            not in self.pending_forecasts
        ):

            raise KeyError(
                "Forecast ID not found "
                "or already processed."
            )

        pending = (
            self.pending_forecasts.pop(
                forecast_id
            )
        )

        observed_state = np.asarray(
            observed_state,
            dtype=float,
        )

        if observed_state.shape != (
            len(self.features),
        ):

            raise ValueError(
                f"Expected observed state "
                f"shape ({len(self.features)},), "
                f"received "
                f"{observed_state.shape}."
            )

        if not np.isfinite(
            observed_state
        ).all():

            raise ValueError(
                "Observed state contains "
                "NaN or Inf."
            )

        observed_kw = float(
            observed_state[
                self.feature_index[
                    "total_power"
                ]
            ]
        )

        # ====================================================
        # ANOMALY
        # ====================================================

        anomaly = (
            self.anomaly_detector
            .evaluate(
                forecast_kw=
                    pending[
                        "forecast_kw"
                    ],

                observed_kw=
                    observed_kw,

                threshold_kw=
                    pending[
                        "threshold_kw"
                    ],
            )
        )

        # Register the error.
        # It is associated with forecast_time + 30 min.
        self.uncertainty.register_error(
            forecast_time=
                pending[
                    "forecast_time"
                ],

            absolute_error=
                anomaly[
                    "absolute_error_kw"
                ],

            threshold_used=
                pending[
                    "threshold_kw"
                ],
        )

        # Make the now-known outcome available
        # to future threshold calculations.
        self.uncertainty.current_threshold(
            pending[
                "outcome_time"
            ]
        )

        observed_components = (
            observed_state[
                self.component_indices
            ]
        )

        root_result = None
        recommendation_result = None

        # ====================================================
        # ROOT CAUSE + RECOMMENDATION
        # ====================================================

        if anomaly[
            "is_anomaly"
        ]:

            root_result = (
                self.root_cause
                .evaluate(
                    now=
                        pending[
                            "outcome_time"
                        ],

                    observed_components=
                        observed_components,

                    total_error_kw=
                        anomaly[
                            "residual_kw"
                        ],

                    update_history=True,
                )
            )

            recommendation_result = (
                self.recommendation
                .build(
                    cause=
                        root_result[
                            "cause"
                        ],

                    residual_kw=
                        anomaly[
                            "residual_kw"
                        ],

                    anomaly_score=
                        anomaly[
                            "anomaly_score"
                        ],

                    state=
                        observed_state,
                )
            )

            alarm_row = {
                "forecast_time":
                    pending[
                        "forecast_time"
                    ],

                "alarm_time":
                    pending[
                        "outcome_time"
                    ],

                "forecast_kw":
                    pending[
                        "forecast_kw"
                    ],

                "observed_kw":
                    observed_kw,

                "residual_kw":
                    anomaly[
                        "residual_kw"
                    ],

                "adaptive_threshold_kw":
                    pending[
                        "threshold_kw"
                    ],

                "anomaly_score":
                    anomaly[
                        "anomaly_score"
                    ],

                "severity":
                    recommendation_result[
                        "severity"
                    ],

                "severity_level":
                    recommendation_result[
                        "severity_level"
                    ],

                "root_cause":
                    root_result[
                        "cause_name"
                    ],

                "root_cause_code":
                    root_result[
                        "cause"
                    ],

                "attribution_strength":
                    root_result[
                        "attribution_strength"
                    ],

                "deviation_direction":
                    anomaly[
                        "direction"
                    ],

                "title":
                    recommendation_result[
                        "title"
                    ],

                "recommendation":
                    recommendation_result[
                        "recommendation"
                    ],

                "action":
                    recommendation_result[
                        "action"
                    ],
            }

            self.alarm_rows.append(
                alarm_row
            )

        else:

            # Normal observations must still
            # update subsystem history.
            self.root_cause.observe(
                now=
                    pending[
                        "outcome_time"
                    ],

                component_values=
                    observed_components,
            )

        return {
            "forecast_id":
                forecast_id,

            "forecast_time":
                pending[
                    "forecast_time"
                ].isoformat(),

            "alarm_time":
                pending[
                    "outcome_time"
                ].isoformat(),

            "anomaly":
                anomaly,

            "root_cause":
                root_result,

            "recommendation":
                recommendation_result,
        }


    # ========================================================
    # INCIDENTS
    # ========================================================

    def build_incidents(
        self
    ):

        if len(
            self.alarm_rows
        ) == 0:

            return (
                pd.DataFrame(),
                pd.DataFrame(),
            )

        alarm_df = pd.DataFrame(
            self.alarm_rows
        )

        return self.incidents.build(
            alarm_df
        )


    # ========================================================
    # STATUS
    # ========================================================

    def status(
        self
    ):

        return {
            "initialized":
                self.initialized,

            "features":
                len(
                    self.features
                ),

            "pending_forecasts":
                len(
                    self.pending_forecasts
                ),

            "alarm_points":
                len(
                    self.alarm_rows
                ),

            "forecast_method":
                "gated_residual_lstm",

            "anomaly_method":
                "adaptive_conformal",

            "root_cause_method":
                "robust_subsystem_attribution",
        }
