from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ForecastRequest(BaseModel):
    week: int = Field(..., ge=1, description="Week number to forecast (may be beyond the training history).")
    center_id: int = Field(..., ge=0)
    meal_id: int = Field(..., ge=0)
    checkout_price: float = Field(..., gt=0)
    base_price: float = Field(..., gt=0)
    emailer_for_promotion: int = Field(0, ge=0, le=1)
    homepage_featured: int = Field(0, ge=0, le=1)

    model_config = {"json_schema_extra": {"example": {
        "week": 146, "center_id": 55, "meal_id": 1885, "checkout_price": 150.0, "base_price": 160.0,
        "emailer_for_promotion": 1, "homepage_featured": 0}}}

    @model_validator(mode="after")
    def _sane_price(self):
        if self.checkout_price > 5 * self.base_price:
            raise ValueError("checkout_price is implausibly high relative to base_price")
        return self


class Economics(BaseModel):
    margin_pct: float = Field(0.40, gt=0, lt=5, description="Profit lost per unmet unit, as a fraction of price.")
    waste_pct: float = Field(0.25, gt=0, lt=5, description="Cost lost per unsold unit, as a fraction of price.")


class ForecastResult(BaseModel):
    week: int
    center_id: int
    meal_id: int
    forecast: float
    lower_80: float
    upper_80: float
    history_points: int
    cold_start: bool
    unknown_center: bool
    unknown_meal: bool


class RecommendRequest(ForecastRequest):
    economics: Economics = Economics()
    service_level: float | None = Field(None, gt=0.5, lt=0.999,
                                        description="Override the cost-optimal service level.")


class RecommendResult(ForecastResult):
    service_level: float
    recommended_stock: int
    expected_waste_cost: float
    expected_lost_margin: float


class BatchRequest(BaseModel):
    items: list[ForecastRequest] = Field(..., min_length=1, max_length=5000)
