
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import EcoTwinEngine


def load_production_engine(
    project_root=None
):

    if project_root is None:

        project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

    project_root = Path(
        project_root
    )

    calibration_path = (
        project_root
        / "models"
        / "ecotwin_runtime_calibration.npz"
    )

    if not calibration_path.exists():

        raise FileNotFoundError(
            "Production calibration artifact "
            "was not found."
        )


    data = np.load(
        calibration_path,
        allow_pickle=False
    )


    # ========================================================
    # RESTORE TIMESTAMPS
    # ========================================================

    calibration_dates = pd.to_datetime(
        data[
            "calibration_forecast_dates_ns"
        ],
        unit="ns"
    )

    component_dates = pd.to_datetime(
        data[
            "component_dates_ns"
        ],
        unit="ns"
    )

    production_start = pd.Timestamp(
        int(
            data[
                "production_start_ns"
            ].item()
        ),
        unit="ns"
    )


    # ========================================================
    # ENGINE
    # ========================================================

    engine = EcoTwinEngine(
        project_root
    )

    initialization = (
        engine.initialize(

            calibration_forecast_dates=
                calibration_dates,

            calibration_absolute_errors=
                data[
                    "calibration_absolute_errors"
                ],

            component_dates=
                component_dates,

            component_values=
                data[
                    "component_values"
                ],

            production_start=
                production_start,
        )
    )


    return (
        engine,
        initialization
    )
