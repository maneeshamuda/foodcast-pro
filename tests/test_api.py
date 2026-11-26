import pytest
from fastapi.testclient import TestClient

from fdf import config

pytestmark = pytest.mark.skipif(not config.ARTIFACT_PATH.exists(), reason="model artifact not trained")

PAYLOAD = {"week": 146, "center_id": 55, "meal_id": 1885, "checkout_price": 150.0, "base_price": 160.0}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_forecast_contract(client):
    r = client.post("/forecast", json=PAYLOAD)
    assert r.status_code == 200
    j = r.json()
    assert j["lower_80"] <= j["forecast"] <= j["upper_80"] and j["forecast"] >= 0


def test_promo_increases_forecast(client):
    base = client.post("/forecast", json=PAYLOAD).json()["forecast"]
    promo = client.post("/forecast", json={**PAYLOAD, "emailer_for_promotion": 1, "homepage_featured": 1}).json()["forecast"]
    assert promo > base


def test_batch_matches_single(client):
    single = client.post("/forecast", json=PAYLOAD).json()["forecast"]
    batch = client.post("/forecast/batch", json={"items": [PAYLOAD, PAYLOAD]}).json()
    assert len(batch) == 2 and batch[0]["forecast"] == pytest.approx(single)


def test_recommend_order_higher_with_higher_service_level(client):
    lo = client.post("/recommend-order", json={**PAYLOAD, "service_level": 0.6}).json()
    hi = client.post("/recommend-order", json={**PAYLOAD, "service_level": 0.95}).json()
    assert hi["recommended_stock"] > lo["recommended_stock"]


@pytest.mark.parametrize("bad", [{"week": 0}, {"checkout_price": -1}, {"emailer_for_promotion": 2}, {"checkout_price": 5000}])
def test_validation_rejects_bad_input(client, bad):
    assert client.post("/forecast", json={**PAYLOAD, **bad}).status_code == 422


def test_unknown_center_flagged(client):
    j = client.post("/forecast", json={**PAYLOAD, "center_id": 99999}).json()
    assert j["unknown_center"] and j["cold_start"]
