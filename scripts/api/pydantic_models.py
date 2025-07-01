
from pydantic import BaseModel

class CustomerData(BaseModel):
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    Amount: float
    ChannelId: str
    ProductCategory: str
    PricingStrategy: str

class PredictionResponse(BaseModel):
    risk_probability: float
