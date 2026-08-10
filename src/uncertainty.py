
from collections import deque

import numpy as np
import pandas as pd


class AdaptiveConformalThreshold:
    """
    EcoTwin adaptive conformal uncertainty engine.

    Final production logic:
    - 96% target coverage
    - 30-day rolling error history
    - 30-minute delayed outcome update
    - anomaly errors are clipped by the threshold
      used when they occurred
    """

    def __init__(
        self,
        coverage=0.96,
        window_days=30,
        horizon_minutes=30
    ):

        self.coverage = float(coverage)
        self.window_days = int(window_days)
        self.horizon_minutes = int(
            horizon_minutes
        )

        self.history = deque()
        self.pending = deque()

        self.initialized = False


    # ========================================================
    # CONFORMAL QUANTILE
    # ========================================================

    @staticmethod
    def conformal_quantile(
        errors,
        coverage
    ):

        errors = np.asarray(
            errors,
            dtype=float
        )

        errors = errors[
            np.isfinite(errors)
        ]

        n = len(errors)

        if n == 0:
            raise ValueError(
                "Calibration history is empty."
            )

        q_level = (
            np.ceil(
                (n + 1)
                * coverage
            )
            / n
        )

        q_level = min(
            q_level,
            1.0
        )

        return float(
            np.quantile(
                errors,
                q_level,
                method="higher"
            )
        )


    # ========================================================
    # INITIAL CALIBRATION
    # ========================================================

    def initialize(
        self,
        forecast_dates,
        absolute_errors,
        production_start
    ):

        """
        Uses the last `window_days` of validation
        outcomes before production_start.
        """

        forecast_dates = pd.to_datetime(
            forecast_dates
        )

        absolute_errors = np.asarray(
            absolute_errors,
            dtype=float
        )

        if (
            len(forecast_dates)
            != len(absolute_errors)
        ):

            raise ValueError(
                "Dates and errors length mismatch."
            )

        production_start = pd.Timestamp(
            production_start
        )

        outcome_dates = (
            forecast_dates
            + pd.Timedelta(
                minutes=self.horizon_minutes
            )
        )

        calibration_start = (
            production_start
            - pd.Timedelta(
                days=self.window_days
            )
        )

        self.history.clear()
        self.pending.clear()

        for date, error in zip(
            outcome_dates,
            absolute_errors
        ):

            if (
                date >= calibration_start
                and
                date <= production_start
                and
                np.isfinite(error)
            ):

                self.history.append(
                    (
                        pd.Timestamp(date),
                        float(error)
                    )
                )

        if len(self.history) == 0:

            raise ValueError(
                "No valid calibration errors "
                "were found."
            )

        self.initialized = True

        return self.current_threshold(
            production_start
        )


    # ========================================================
    # HISTORY MAINTENANCE
    # ========================================================

    def _prune_history(
        self,
        now
    ):

        now = pd.Timestamp(
            now
        )

        cutoff = (
            now
            - pd.Timedelta(
                days=self.window_days
            )
        )

        while (
            len(self.history) > 0
            and
            self.history[0][0]
            < cutoff
        ):

            self.history.popleft()


    def _flush_pending(
        self,
        now
    ):

        """
        Only forecast errors whose outcomes are
        already known may enter history.
        """

        now = pd.Timestamp(
            now
        )

        while (
            len(self.pending) > 0
            and
            self.pending[0][0]
            <= now
        ):

            (
                outcome_time,
                error,
                threshold_used
            ) = self.pending.popleft()

            clipped_error = min(
                float(error),
                float(threshold_used)
            )

            self.history.append(
                (
                    outcome_time,
                    clipped_error
                )
            )


    # ========================================================
    # THRESHOLD
    # ========================================================

    def current_threshold(
        self,
        now
    ):

        if not self.initialized:
            raise RuntimeError(
                "Threshold engine is not initialized."
            )

        now = pd.Timestamp(
            now
        )

        self._flush_pending(
            now
        )

        self._prune_history(
            now
        )

        errors = [
            item[1]
            for item in self.history
        ]

        return self.conformal_quantile(
            errors,
            self.coverage
        )


    # ========================================================
    # REGISTER FORECAST OUTCOME
    # ========================================================

    def register_error(
        self,
        forecast_time,
        absolute_error,
        threshold_used
    ):

        """
        Error becomes available only after the
        forecast horizon has elapsed.
        """

        forecast_time = pd.Timestamp(
            forecast_time
        )

        outcome_time = (
            forecast_time
            + pd.Timedelta(
                minutes=self.horizon_minutes
            )
        )

        self.pending.append(
            (
                outcome_time,
                float(absolute_error),
                float(threshold_used)
            )
        )
