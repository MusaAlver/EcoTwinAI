
import numpy as np


class EcoTwinRecommendationEngine:

    def __init__(self, feature_names=None):

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else []
        )

        self.feature_index = {
            name: i
            for i, name in enumerate(
                self.feature_names
            )
        }


    # ========================================================
    # FEATURE HELPER
    # ========================================================

    def get_feature(
        self,
        state,
        name,
        default=np.nan
    ):

        if name not in self.feature_index:
            return default

        state = np.asarray(
            state,
            dtype=float
        )

        return float(
            state[
                self.feature_index[name]
            ]
        )


    # ========================================================
    # SEVERITY
    # ========================================================

    @staticmethod
    def get_severity(
        anomaly_score
    ):

        anomaly_score = float(
            anomaly_score
        )

        if anomaly_score < 1.0:
            return "NORMAL", 0

        if anomaly_score < 1.5:
            return "WARNING", 1

        if anomaly_score < 2.5:
            return "HIGH", 2

        return "CRITICAL", 3


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    def build(
        self,
        cause,
        residual_kw,
        anomaly_score,
        state=None
    ):

        residual_kw = float(
            residual_kw
        )

        anomaly_score = float(
            anomaly_score
        )

        severity, severity_level = (
            self.get_severity(
                anomaly_score
            )
        )

        direction = (
            "HIGH_CONSUMPTION"
            if residual_kw > 0
            else
            "LOW_CONSUMPTION"
        )


        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        if state is None:

            indoor_temp = np.nan
            outdoor_temp = np.nan
            humidity = np.nan

        else:

            indoor_temp = self.get_feature(
                state,
                "indoor_temp_avg"
            )

            outdoor_temp = self.get_feature(
                state,
                "outdoor_temp_avg"
            )

            humidity = self.get_feature(
                state,
                "relative_humidity_set_1"
            )


        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        if severity == "NORMAL":

            return {
                "severity":
                    severity,

                "severity_level":
                    severity_level,

                "direction":
                    direction,

                "title":
                    "Normal operation",

                "recommendation":
                    "No intervention required.",

                "action":
                    "Continue monitoring.",

                "indoor_temp":
                    indoor_temp,

                "outdoor_temp":
                    outdoor_temp,

                "humidity":
                    humidity,
            }


        # ----------------------------------------------------
        # HVAC
        # ----------------------------------------------------

        if cause in [
            "hvac_N",
            "hvac_S"
        ]:

            zone = (
                "North"
                if cause == "hvac_N"
                else "South"
            )

            if residual_kw > 0:

                title = (
                    f"Unexpected HVAC consumption "
                    f"— {zone} zone"
                )

                recommendation = (
                    f"Inspect the {zone} HVAC operating "
                    f"state, schedule and temperature "
                    f"setpoint."
                )

                if (
                    np.isfinite(indoor_temp)
                    and
                    np.isfinite(outdoor_temp)
                ):

                    temp_gap = abs(
                        indoor_temp
                        -
                        outdoor_temp
                    )

                    if temp_gap > 8:

                        recommendation += (
                            " A large indoor–outdoor "
                            "temperature difference is "
                            "also present."
                        )

            else:

                title = (
                    f"HVAC consumption below expected "
                    f"— {zone} zone"
                )

                recommendation = (
                    f"Check whether the {zone} HVAC "
                    f"system is operating according "
                    f"to schedule and setpoint."
                )

            action = (
                f"Review HVAC {zone} operation."
            )


        # ----------------------------------------------------
        # MELS
        # ----------------------------------------------------

        elif cause in [
            "mels_N",
            "mels_S"
        ]:

            zone = (
                "North"
                if cause == "mels_N"
                else "South"
            )

            if residual_kw > 0:

                title = (
                    f"Unexpected equipment load "
                    f"— {zone} zone"
                )

                recommendation = (
                    f"Review plug-load and equipment "
                    f"activity in the {zone} zone. "
                    f"Check for devices operating "
                    f"outside the expected schedule."
                )

            else:

                title = (
                    f"Equipment load below expected "
                    f"— {zone} zone"
                )

                recommendation = (
                    f"Check whether equipment in the "
                    f"{zone} zone is inactive or "
                    f"offline unexpectedly."
                )

            action = (
                f"Review MELS {zone} loads."
            )


        # ----------------------------------------------------
        # LIGHTING
        # ----------------------------------------------------

        elif cause == "lig_S":

            if residual_kw > 0:

                title = (
                    "Unexpected lighting consumption"
                )

                recommendation = (
                    "Check lighting schedules and "
                    "whether lights are active outside "
                    "the expected operating period."
                )

            else:

                title = (
                    "Lighting consumption below expected"
                )

                recommendation = (
                    "Check lighting system availability "
                    "and scheduled operating state."
                )

            action = (
                "Review lighting operation."
            )


        # ----------------------------------------------------
        # UNEXPLAINED
        # ----------------------------------------------------

        else:

            title = (
                "Unexplained energy deviation"
            )

            if residual_kw > 0:

                recommendation = (
                    "The observed increase cannot be "
                    "explained by the monitored HVAC, "
                    "MELS or lighting subsystems. "
                    "Verify meter consistency and "
                    "inspect unmonitored electrical loads."
                )

            else:

                recommendation = (
                    "The observed reduction cannot be "
                    "explained by the monitored HVAC, "
                    "MELS or lighting subsystems. "
                    "Verify meter and sensor consistency "
                    "and check for unexpected shutdowns."
                )

            action = (
                "Inspect metering and unmonitored loads."
            )


        return {
            "severity":
                severity,

            "severity_level":
                severity_level,

            "direction":
                direction,

            "title":
                title,

            "recommendation":
                recommendation,

            "action":
                action,

            "indoor_temp":
                indoor_temp,

            "outdoor_temp":
                outdoor_temp,

            "humidity":
                humidity,
        }
