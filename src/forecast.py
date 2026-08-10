
from pathlib import Path
import json

import joblib
import numpy as np
import tensorflow as tf


class EcoTwinForecaster:
    """
    EcoTwin AI 30-minute energy forecasting engine.

    Final method:
        Persistence + gated LSTM residual correction
    """

    def __init__(self, project_root=None):

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

        # ----------------------------------------
        # Artifact paths
        # ----------------------------------------

        self.model_path = (
            self.model_dir
            / "ecotwin_30m_residual_lstm.keras"
        )

        self.x_scaler_path = (
            self.model_dir
            / "ecotwin_30m_x_scaler.pkl"
        )

        self.residual_scaler_path = (
            self.model_dir
            / "ecotwin_30m_residual_scaler.pkl"
        )

        self.feature_path = (
            self.model_dir
            / "ecotwin_30m_features.json"
        )

        self.config_path = (
            self.model_dir
            / "ecotwin_forecast_config.json"
        )

        # ----------------------------------------
        # Load artifacts
        # ----------------------------------------

        self.model = (
            tf.keras.models.load_model(
                self.model_path,
                compile=False
            )
        )

        self.x_scaler = joblib.load(
            self.x_scaler_path
        )

        self.residual_scaler = joblib.load(
            self.residual_scaler_path
        )

        with open(
            self.feature_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.features = json.load(f)

        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.config = json.load(f)

        # ----------------------------------------
        # Production contract
        # ----------------------------------------

        forecast_cfg = (
            self.config["forecast_30m"]
        )

        self.history_timesteps = int(
            forecast_cfg[
                "history_timesteps"
            ]
        )

        self.feature_count = int(
            forecast_cfg[
                "feature_count"
            ]
        )

        self.gate_lambda = float(
            forecast_cfg[
                "gate_lambda"
            ]
        )

        self.total_power_index = (
            self.features.index(
                "total_power"
            )
        )

        self._validate_artifacts()


    # ========================================================
    # ARTIFACT VALIDATION
    # ========================================================

    def _validate_artifacts(self):

        if (
            len(self.features)
            != self.feature_count
        ):

            raise ValueError(
                "Feature contract mismatch."
            )

        if (
            self.x_scaler.n_features_in_
            != self.feature_count
        ):

            raise ValueError(
                "X scaler feature count mismatch."
            )

        if (
            self.model.input_shape[1:]
            != (
                self.history_timesteps,
                self.feature_count
            )
        ):

            raise ValueError(
                "Model input contract mismatch."
            )

        if (
            self.model.output_shape[-1]
            != 1
        ):

            raise ValueError(
                "Unexpected model output."
            )


    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def _prepare_window(
        self,
        raw_window
    ):

        window = np.asarray(
            raw_window,
            dtype=np.float32
        )

        expected_shape = (
            self.history_timesteps,
            self.feature_count
        )

        if (
            window.shape
            != expected_shape
        ):

            raise ValueError(
                f"Expected raw window "
                f"{expected_shape}, "
                f"received {window.shape}."
            )

        if not np.isfinite(
            window
        ).all():

            raise ValueError(
                "Input contains NaN or Inf."
            )

        # Scale timestep by timestep
        scaled = (
            self.x_scaler
            .transform(
                window
            )
            .astype(
                np.float32
            )
        )

        return scaled[
            np.newaxis,
            ...
        ]


    # ========================================================
    # 30-MINUTE FORECAST
    # ========================================================

    def predict_30m(
        self,
        raw_window,
        persistence_kw=None
    ):

        """
        Parameters
        ----------
        raw_window:
            Shape (16, 23), unscaled features.

        persistence_kw:
            Optional current power value.

            If omitted, the latest total_power
            value in raw_window is used.

        Returns
        -------
        dict
        """

        raw_window = np.asarray(
            raw_window,
            dtype=np.float32
        )

        scaled_window = (
            self._prepare_window(
                raw_window
            )
        )

        # ----------------------------------------
        # Persistence baseline
        # ----------------------------------------

        if persistence_kw is None:

            persistence_kw = float(
                raw_window[
                    -1,
                    self.total_power_index
                ]
            )

        else:

            persistence_kw = float(
                persistence_kw
            )

        # ----------------------------------------
        # LSTM residual prediction
        # ----------------------------------------

        scaled_residual = (
            self.model.predict(
                scaled_window,
                verbose=0
            )
        )

        residual_kw = float(
            self.residual_scaler
            .inverse_transform(
                scaled_residual
            )
            .reshape(-1)[0]
        )

        # ----------------------------------------
        # Gated final forecast
        # ----------------------------------------

        correction_kw = (
            self.gate_lambda
            *
            residual_kw
        )

        forecast_kw = (
            persistence_kw
            +
            correction_kw
        )

        return {

            "forecast_horizon_minutes":
                30,

            "forecast_kw":
                float(forecast_kw),

            "persistence_kw":
                persistence_kw,

            "predicted_residual_kw":
                residual_kw,

            "gate_lambda":
                self.gate_lambda,

            "applied_correction_kw":
                float(correction_kw),

            "method":
                "gated_residual_lstm"
        }
