
import numpy as np
import pandas as pd


class EcoTwinIncidentEngine:
    """
    EcoTwin operational incident aggregation engine.

    Alarm points are merged into incidents.

    A new incident starts when:
    1) Gap from previous alarm > max_gap_minutes
    2) Consumption direction changes
    """

    SEVERITY_RANK = {
        "WARNING": 1,
        "HIGH": 2,
        "CRITICAL": 3,
    }

    def __init__(
        self,
        max_gap_minutes=30,
        observation_minutes=15,
    ):

        self.max_gap_minutes = int(
            max_gap_minutes
        )

        self.observation_minutes = int(
            observation_minutes
        )



    @staticmethod
    def _validate_columns(df):

        required = {
            "alarm_time",
            "severity",
            "anomaly_score",
            "root_cause",
            "residual_kw",
            "observed_kw",
            "forecast_kw",
        }

        missing = (
            required
            -
            set(df.columns)
        )

        if missing:

            raise ValueError(
                "Missing incident columns: "
                + ", ".join(
                    sorted(missing)
                )
            )



    @staticmethod
    def dominant_incident_cause(
        group
    ):

        cause_scores = {}

        for _, row in group.iterrows():

            cause = row[
                "root_cause"
            ]

            anomaly_score = float(
                row[
                    "anomaly_score"
                ]
            )

            strength = row.get(
                "attribution_strength",
                np.nan
            )

            if (
                cause == "UNEXPLAINED LOAD"
                or
                pd.isna(strength)
            ):

                weight = (
                    anomaly_score
                )

            else:

                weight = (
                    anomaly_score
                    *
                    float(strength)
                    /
                    100.0
                )

            cause_scores[
                cause
            ] = (
                cause_scores.get(
                    cause,
                    0.0
                )
                +
                weight
            )

        return max(
            cause_scores,
            key=cause_scores.get
        )



    def build(
        self,
        alarm_df
    ):

        if not isinstance(
            alarm_df,
            pd.DataFrame
        ):

            raise TypeError(
                "alarm_df must be "
                "a pandas DataFrame."
            )

        self._validate_columns(
            alarm_df
        )


        if len(alarm_df) == 0:

            empty_source = (
                alarm_df.copy()
            )

            empty_incidents = (
                pd.DataFrame()
            )

            return (
                empty_source,
                empty_incidents
            )

        incident_source = (
            alarm_df.copy()
        )

        incident_source[
            "alarm_time"
        ] = pd.to_datetime(
            incident_source[
                "alarm_time"
            ]
        )


        if (
            "attribution_strength"
            not in incident_source.columns
        ):

            incident_source[
                "attribution_strength"
            ] = np.nan

        incident_source[
            "attribution_strength"
        ] = pd.to_numeric(
            incident_source[
                "attribution_strength"
            ],
            errors="coerce"
        )


        if (
            "deviation_direction"
            not in incident_source.columns
        ):

            incident_source[
                "deviation_direction"
            ] = np.where(
                incident_source[
                    "residual_kw"
                ]
                >= 0,
                "HIGH_CONSUMPTION",
                "LOW_CONSUMPTION",
            )

        incident_source = (
            incident_source
            .sort_values(
                "alarm_time"
            )
            .reset_index(
                drop=True
            )
        )


        incident_source[
            "severity_rank"
        ] = (
            incident_source[
                "severity"
            ]
            .map(
                self.SEVERITY_RANK
            )
            .fillna(0)
            .astype(int)
        )


        time_gap = (
            incident_source[
                "alarm_time"
            ]
            .diff()
        )

        direction_changed = (
            incident_source[
                "deviation_direction"
            ]
            !=
            incident_source[
                "deviation_direction"
            ].shift(1)
        )

        new_incident = (
            time_gap.isna()
            |
            (
                time_gap
                >
                pd.Timedelta(
                    minutes=
                        self.max_gap_minutes
                )
            )
            |
            direction_changed
        )

        incident_source[
            "incident_id"
        ] = (
            new_incident
            .cumsum()
            .astype(int)
        )


        incident_rows = []

        for (
            incident_id,
            group
        ) in incident_source.groupby(
            "incident_id"
        ):

            group = (
                group
                .sort_values(
                    "alarm_time"
                )
            )

            peak_row = (
                group
                .sort_values(
                    [
                        "severity_rank",
                        "anomaly_score",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
                .iloc[0]
            )

            dominant_cause = (
                self
                .dominant_incident_cause(
                    group
                )
            )

            start_time = (
                group[
                    "alarm_time"
                ].min()
            )

            end_time = (
                group[
                    "alarm_time"
                ].max()
            )

            duration_minutes = (
                (
                    end_time
                    -
                    start_time
                )
                .total_seconds()
                /
                60.0
                +
                self.observation_minutes
            )

            valid_strength = (
                group[
                    "attribution_strength"
                ]
                .dropna()
            )

            if len(
                valid_strength
            ) > 0:

                mean_strength = float(
                    valid_strength.mean()
                )

                max_strength = float(
                    valid_strength.max()
                )

            else:

                mean_strength = np.nan
                max_strength = np.nan

            incident_rows.append(
                {
                    "incident_id":
                        int(
                            incident_id
                        ),

                    "start_time":
                        start_time,

                    "end_time":
                        end_time,

                    "duration_minutes":
                        float(
                            duration_minutes
                        ),

                    "alarm_points":
                        int(
                            len(group)
                        ),

                    "direction":
                        group[
                            "deviation_direction"
                        ].iloc[0],

                    "peak_severity":
                        peak_row[
                            "severity"
                        ],

                    "peak_anomaly_score":
                        float(
                            peak_row[
                                "anomaly_score"
                            ]
                        ),

                    "dominant_root_cause":
                        dominant_cause,

                    "mean_attribution_strength":
                        mean_strength,

                    "max_attribution_strength":
                        max_strength,

                    "max_positive_deviation_kw":
                        float(
                            group[
                                "residual_kw"
                            ].max()
                        ),

                    "max_negative_deviation_kw":
                        float(
                            group[
                                "residual_kw"
                            ].min()
                        ),

                    "mean_abs_deviation_kw":
                        float(
                            group[
                                "residual_kw"
                            ]
                            .abs()
                            .mean()
                        ),

                    "peak_observed_kw":
                        float(
                            group[
                                "observed_kw"
                            ].max()
                        ),

                    "min_observed_kw":
                        float(
                            group[
                                "observed_kw"
                            ].min()
                        ),

                    "peak_forecast_kw":
                        float(
                            group[
                                "forecast_kw"
                            ].max()
                        ),
                }
            )

        incident_df = pd.DataFrame(
            incident_rows
        )

        return (
            incident_source,
            incident_df
        )



    @staticmethod
    def summarize(
        incident_source,
        incident_df
    ):

        alarm_points = len(
            incident_source
        )

        incident_count = len(
            incident_df
        )

        compression = (
            (
                1
                -
                incident_count
                /
                alarm_points
            )
            * 100
            if alarm_points > 0
            else 0.0
        )

        explained = (
            incident_df[
                incident_df[
                    "dominant_root_cause"
                ]
                !=
                "UNEXPLAINED LOAD"
            ]
            if incident_count > 0
            else pd.DataFrame()
        )

        if (
            incident_count > 0
            and
            "mean_attribution_strength"
            in explained.columns
        ):

            strong = (
                explained[
                    explained[
                        "mean_attribution_strength"
                    ]
                    >= 60
                ]
            )

        else:

            strong = (
                pd.DataFrame()
            )

        strong_rate = (
            len(strong)
            /
            len(explained)
            * 100
            if len(explained) > 0
            else 0.0
        )

        return {
            "alarm_points":
                alarm_points,

            "incident_count":
                incident_count,

            "compression_percent":
                float(
                    compression
                ),

            "explained_incidents":
                len(
                    explained
                ),

            "strong_incidents":
                len(
                    strong
                ),

            "strong_attribution_rate_percent":
                float(
                    strong_rate
                ),
        }
