
from collections import defaultdict, deque

import numpy as np
import pandas as pd


COMPONENTS = [
    "hvac_N",
    "hvac_S",
    "mels_N",
    "mels_S",
    "lig_S",
]


COMPONENT_NAMES = {
    "hvac_N": "HVAC NORTH",
    "hvac_S": "HVAC SOUTH",
    "mels_N": "MELS NORTH",
    "mels_S": "MELS SOUTH",
    "lig_S": "LIGHTING",
}


class EcoTwinRootCauseEngine:
    """
    EcoTwin subsystem root-cause attribution engine.

    Expected subsystem behaviour is estimated from:
    - same weekday/weekend category
    - same hour
    - same minute
    - previous 28 days

    Robust statistics:
    - median
    - MAD
    """

    def __init__(
        self,
        window_days=28,
        minimum_scale=0.25,
        local_scale_floor_ratio=0.20,
    ):

        self.window_days = int(
            window_days
        )

        self.minimum_scale = float(
            minimum_scale
        )

        self.local_scale_floor_ratio = float(
            local_scale_floor_ratio
        )

        self.slot_history = defaultdict(
            deque
        )

        self.global_median = None
        self.global_scale = None

        self.initialized = False


    # ========================================================
    # TIME SLOT
    # ========================================================

    @staticmethod
    def slot_key(date):

        date = pd.Timestamp(
            date
        )

        return (
            int(
                date.dayofweek >= 5
            ),
            int(date.hour),
            int(date.minute),
        )


    # ========================================================
    # INITIAL CALIBRATION
    # ========================================================

    def initialize(
        self,
        dates,
        component_values,
        production_start,
    ):

        dates = pd.to_datetime(
            dates
        )

        component_values = np.asarray(
            component_values,
            dtype=float,
        )

        if (
            component_values.ndim != 2
            or
            component_values.shape[1]
            != len(COMPONENTS)
        ):

            raise ValueError(
                "Expected component matrix "
                "with shape (n, 5)."
            )

        if (
            len(dates)
            != len(component_values)
        ):

            raise ValueError(
                "Dates and component values "
                "length mismatch."
            )

        production_start = pd.Timestamp(
            production_start
        )

        history_start = (
            production_start
            - pd.Timedelta(
                days=self.window_days
            )
        )

        # ----------------------------------------------------
        # Global robust fallback
        # ----------------------------------------------------

        self.global_median = np.nanmedian(
            component_values,
            axis=0,
        )

        global_mad = np.nanmedian(
            np.abs(
                component_values
                - self.global_median
            ),
            axis=0,
        )

        self.global_scale = (
            1.4826
            * global_mad
        )

        self.global_scale = np.maximum(
            self.global_scale,
            self.minimum_scale,
        )

        # ----------------------------------------------------
        # Last 28 days → slot history
        # ----------------------------------------------------

        self.slot_history.clear()

        for date, values in zip(
            dates,
            component_values,
        ):

            if (
                date >= history_start
                and
                date < production_start
            ):

                key = self.slot_key(
                    date
                )

                self.slot_history[
                    key
                ].append(
                    (
                        pd.Timestamp(date),
                        values.copy(),
                    )
                )

        self.initialized = True


    # ========================================================
    # EXPECTED SUBSYSTEM STATE
    # ========================================================

    def expected_state(
        self,
        now,
    ):

        if not self.initialized:

            raise RuntimeError(
                "Root-cause engine "
                "is not initialized."
            )

        now = pd.Timestamp(
            now
        )

        key = self.slot_key(
            now
        )

        history = self.slot_history[
            key
        ]

        cutoff = (
            now
            - pd.Timedelta(
                days=self.window_days
            )
        )

        while (
            len(history) > 0
            and
            history[0][0] < cutoff
        ):

            history.popleft()

        # ----------------------------------------------------
        # Local robust expectation
        # ----------------------------------------------------

        if len(history) >= 3:

            matrix = np.vstack(
                [
                    item[1]
                    for item in history
                ]
            )

            local_median = np.nanmedian(
                matrix,
                axis=0,
            )

            local_mad = np.nanmedian(
                np.abs(
                    matrix
                    - local_median
                ),
                axis=0,
            )

            local_scale = (
                1.4826
                * local_mad
            )

            local_scale = np.maximum(
                local_scale,
                self.global_scale
                *
                self.local_scale_floor_ratio,
            )

            local_scale = np.maximum(
                local_scale,
                self.minimum_scale,
            )

        else:

            local_median = (
                self.global_median.copy()
            )

            local_scale = (
                self.global_scale.copy()
            )

        return (
            local_median,
            local_scale,
        )


    # ========================================================
    # ADD REAL BUILDING STATE
    # ========================================================

    def observe(
        self,
        now,
        component_values,
    ):

        values = np.asarray(
            component_values,
            dtype=float,
        )

        if values.shape != (
            len(COMPONENTS),
        ):

            raise ValueError(
                "Expected 5 subsystem values."
            )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                "Subsystem state contains "
                "NaN or Inf."
            )

        now = pd.Timestamp(
            now
        )

        key = self.slot_key(
            now
        )

        self.slot_history[
            key
        ].append(
            (
                now,
                values.copy(),
            )
        )


    # ========================================================
    # ROOT CAUSE DIAGNOSIS
    # ========================================================

    def diagnose(
        self,
        observed_components,
        expected_components,
        component_scales,
        total_error_kw,
    ):

        observed = np.asarray(
            observed_components,
            dtype=float,
        )

        expected = np.asarray(
            expected_components,
            dtype=float,
        )

        scales = np.asarray(
            component_scales,
            dtype=float,
        )

        if not (
            observed.shape
            ==
            expected.shape
            ==
            scales.shape
            ==
            (len(COMPONENTS),)
        ):

            raise ValueError(
                "Subsystem vectors must "
                "all have shape (5,)."
            )

        total_error_kw = float(
            total_error_kw
        )

        # Positive total error:
        # inspect positive subsystem deviation.
        #
        # Negative total error:
        # inspect negative subsystem deviation.

        direction = (
            1.0
            if total_error_kw >= 0
            else -1.0
        )

        delta = (
            observed
            - expected
        )

        directional_delta = np.maximum(
            direction * delta,
            0.0,
        )

        robust_z = (
            directional_delta
            /
            np.maximum(
                scales,
                self.minimum_scale,
            )
        )

        # kW contribution + unusualness
        score = (
            directional_delta
            *
            (
                1.0
                +
                np.clip(
                    robust_z,
                    0,
                    5,
                )
                / 5.0
            )
        )

        total_score = float(
            np.sum(score)
        )

        total_deviation = abs(
            total_error_kw
        )

        # ----------------------------------------------------
        # No monitored subsystem sufficiently explains event
        # ----------------------------------------------------

        if (
            total_score <= 1e-8
            or
            np.max(
                directional_delta
            )
            <
            max(
                0.75,
                total_deviation
                * 0.10,
            )
        ):

            return {

                "cause":
                    "UNEXPLAINED_LOAD",

                "cause_name":
                    "UNEXPLAINED LOAD",

                "attribution_strength":
                    np.nan,

                "delta":
                    delta,

                "robust_z":
                    robust_z,

                "scores":
                    score,
            }

        best_idx = int(
            np.argmax(
                score
            )
        )

        attribution_strength = (
            score[
                best_idx
            ]
            /
            max(
                np.sum(score),
                1e-8,
            )
            * 100
        )

        cause = COMPONENTS[
            best_idx
        ]

        return {

            "cause":
                cause,

            "cause_name":
                COMPONENT_NAMES[
                    cause
                ],

            "attribution_strength":
                float(
                    attribution_strength
                ),

            "delta":
                delta,

            "robust_z":
                robust_z,

            "scores":
                score,
        }


    # ========================================================
    # COMPLETE REAL-TIME STEP
    # ========================================================

    def evaluate(
        self,
        now,
        observed_components,
        total_error_kw,
        update_history=True,
    ):

        expected, scales = (
            self.expected_state(
                now
            )
        )

        diagnosis = self.diagnose(
            observed_components=
                observed_components,

            expected_components=
                expected,

            component_scales=
                scales,

            total_error_kw=
                total_error_kw,
        )

        diagnosis[
            "expected_components"
        ] = expected

        diagnosis[
            "component_scales"
        ] = scales

        if update_history:

            self.observe(
                now,
                observed_components,
            )

        return diagnosis
