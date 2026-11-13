"""Public JSON contract, respecting original Wine Quality field names."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class WineFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False,
                              json_schema_extra={"examples": [{
                                  "fixed acidity": 7.0, "volatile acidity": 0.27, "citric acid": 0.36,
                                  "residual sugar": 20.7, "chlorides": 0.045,
                                  "free sulfur dioxide": 45.0, "total sulfur dioxide": 170.0,
                                  "density": 1.001, "pH": 3.0, "sulphates": 0.45,
                                  "alcohol": 8.8, "тип_вина": "белое",
                              }]})

    fixed_acidity: float = Field(ge=0, le=30, alias="fixed acidity")
    volatile_acidity: float = Field(ge=0, le=3, alias="volatile acidity")
    citric_acid: float = Field(ge=0, le=3, alias="citric acid")
    residual_sugar: float = Field(ge=0, le=150, alias="residual sugar")
    chlorides: float = Field(ge=0, le=2, alias="chlorides")
    free_sulfur_dioxide: float = Field(ge=0, le=500, alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(ge=0, le=1000, alias="total sulfur dioxide")
    density: float = Field(ge=0.9, le=1.1)
    ph: float = Field(ge=0, le=14, alias="pH")
    sulphates: float = Field(ge=0, le=5)
    alcohol: float = Field(ge=0, le=25)
    wine_type: Literal["красное", "белое"] = Field(alias="тип_вина")

    @model_validator(mode="after")
    def check_so2(self):
        if self.free_sulfur_dioxide > self.total_sulfur_dioxide:
            raise ValueError("free sulfur dioxide не может быть больше total sulfur dioxide")
        return self


class Prediction(BaseModel):
    predicted_class: Literal[0, 1]
    label: Literal["хорошее", "остальное"]
    probability: float = Field(ge=0, le=1)
    threshold: float
    model_version: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str


class ModelInfo(BaseModel):
    model_version: str
    data_version: str
    trained_at: str
    dataset: str
    algorithm: str
    features: list[str]
    target: str
    threshold: float
    demo_mode: bool
    metrics: dict
    data_counts: dict
