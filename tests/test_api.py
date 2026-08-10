
from fastapi.testclient import (
    TestClient
)

from src.api import app


def test_health():

    with TestClient(app) as client:

        response = client.get(
            "/health"
        )

        assert (
            response.status_code
            ==
            200
        )

        data = response.json()

        assert (
            data["status"]
            ==
            "ok"
        )

        assert (
            data[
                "engine_loaded"
            ]
            is True
        )

        assert (
            data[
                "initialized"
            ]
            is True
        )


def test_status():

    with TestClient(app) as client:

        response = client.get(
            "/status"
        )

        assert (
            response.status_code
            ==
            200
        )

        data = response.json()

        assert (
            data[
                "forecast_method"
            ]
            ==
            "gated_residual_lstm"
        )

        assert (
            data[
                "anomaly_method"
            ]
            ==
            "adaptive_conformal"
        )


def test_incidents_empty_at_start():

    with TestClient(app) as client:

        response = client.get(
            "/incidents"
        )

        assert (
            response.status_code
            ==
            200
        )

        data = response.json()

        assert (
            data["count"]
            ==
            0
        )

        assert (
            data["incidents"]
            ==
            []
        )
