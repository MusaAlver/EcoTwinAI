
import numpy as np


class EcoTwinAnomalyDetector:
    """
    EcoTwin forecast-residual anomaly detector.
    """

    def evaluate(
        self,
        forecast_kw,
        observed_kw,
        threshold_kw
    ):

        forecast_kw = float(
            forecast_kw
        )

        observed_kw = float(
            observed_kw
        )

        threshold_kw = float(
            threshold_kw
        )

        if (
            not np.isfinite(forecast_kw)
            or
            not np.isfinite(observed_kw)
            or
            not np.isfinite(threshold_kw)
        ):

            raise ValueError(
                "Inputs must be finite."
            )

        if threshold_kw <= 0:

            raise ValueError(
                "Threshold must be positive."
            )

        residual_kw = (
            observed_kw
            - forecast_kw
        )

        absolute_error_kw = abs(
            residual_kw
        )

        anomaly_score = (
            absolute_error_kw
            /
            max(
                threshold_kw,
                1e-6
            )
        )

        is_anomaly = (
            absolute_error_kw
            >
            threshold_kw
        )

        direction = (
            "HIGH_CONSUMPTION"
            if residual_kw > 0
            else
            "LOW_CONSUMPTION"
        )

        return {

            "forecast_kw":
                forecast_kw,

            "observed_kw":
                observed_kw,

            "residual_kw":
                float(
                    residual_kw
                ),

            "absolute_error_kw":
                float(
                    absolute_error_kw
                ),

            "threshold_kw":
                threshold_kw,

            "anomaly_score":
                float(
                    anomaly_score
                ),

            "is_anomaly":
                bool(
                    is_anomaly
                ),

            "direction":
                direction
        }
