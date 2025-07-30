from typing import List
from pydantic import BaseModel


class DataInstance(BaseModel):
    pressure: List[float]
    flow: List[float]