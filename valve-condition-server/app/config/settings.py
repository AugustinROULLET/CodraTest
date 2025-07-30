
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    BASE_DIRECTORY: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIRECTORY: Path = BASE_DIRECTORY / "data"
    MODEL_DIRECTORY: Path = BASE_DIRECTORY / "prediction_model"
    STATISTICS_FILE: Path = MODEL_DIRECTORY / "statistics.json"
    NUMBER_EPOCHS: int = 15



settings = Settings()