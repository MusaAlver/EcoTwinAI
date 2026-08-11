
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from fastapi.encoders import jsonable_encoder

from fastapi import (
    FastAPI,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
)

from .runtime import (
    load_production_engine
)


runtime = {
    "engine": None,
    "startup": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    engine, startup = (
        load_production_engine(
            project_root
        )
    )

    runtime["engine"] = engine
    runtime["startup"] = startup

    print(
        "EcoTwin production engine loaded ✅"
    )

    yield

    runtime["engine"] = None
    runtime["startup"] = None


app = FastAPI(

    title=
        "EcoTwin AI API",

    description=
        (
            "Smart-building energy forecasting, "
            "adaptive anomaly detection, "
            "root-cause attribution and "
            "incident intelligence."
        ),

    version=
        "1.0.0",

    lifespan=
        lifespan,
)


class ForecastRequest(BaseModel):

    forecast_time: str

    raw_window: list[
        list[float]
    ]

    persistence_kw: (
        float | None
    ) = None


class OutcomeRequest(BaseModel):

    forecast_id: str

    observed_state: list[float]


@app.get(
    "/health"
)
def health():

    engine = runtime[
        "engine"
    ]

    return {

        "status":
            "ok",

        "service":
            "EcoTwin AI",

        "version":
            "1.0.0",

        "engine_loaded":
            engine is not None,

        "initialized":
            (
                engine.initialized
                if engine is not None
                else False
            ),
    }


@app.get(
    "/status"
)
def status():

    engine = runtime[
        "engine"
    ]

    if engine is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "EcoTwin engine "
                "is not available."
            ),
        )

    return engine.status()


@app.post(
    "/forecast"
)
def forecast(
    request: ForecastRequest
):

    engine = runtime[
        "engine"
    ]

    if engine is None:

        raise HTTPException(
            status_code=503,
            detail="Engine unavailable.",
        )

    try:

        raw_window = np.asarray(
            request.raw_window,
            dtype=np.float32,
        )

        result = (
            engine.create_forecast(

                forecast_time=
                    request.forecast_time,

                raw_window=
                    raw_window,

                persistence_kw=
                    request.persistence_kw,
            )
        )

        return result

    except (
        ValueError,
        RuntimeError,
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post(
    "/outcome"
)
def outcome(
    request: OutcomeRequest
):

    engine = runtime[
        "engine"
    ]

    if engine is None:

        raise HTTPException(
            status_code=503,
            detail="Engine unavailable.",
        )

    try:

        result = (
            engine.process_outcome(

                forecast_id=
                    request.forecast_id,

                observed_state=
                    request.observed_state,
            )
        )

        return jsonable_encoder(
            result,
            custom_encoder={
                np.ndarray: lambda value: value.tolist(),
                np.generic: lambda value: value.item(),
            },
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except (
        ValueError,
        RuntimeError,
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get(
    "/incidents"
)
def incidents():

    engine = runtime[
        "engine"
    ]

    if engine is None:

        raise HTTPException(
            status_code=503,
            detail="Engine unavailable.",
        )

    _, incident_df = (
        engine.build_incidents()
    )

    if len(
        incident_df
    ) == 0:

        return {
            "count": 0,
            "incidents": [],
        }

    output = (
        incident_df
        .copy()
    )

    for column in [
        "start_time",
        "end_time",
    ]:

        if column in output.columns:

            output[
                column
            ] = (
                output[
                    column
                ]
                .astype(str)
            )

    output = output.replace(
        {
            np.nan: None
        }
    )

    return {

        "count":
            len(output),

        "incidents":
            output.to_dict(
                orient="records"
            ),
    }
