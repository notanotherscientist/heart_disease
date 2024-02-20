import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Dict, List
load_dotenv()


class Settings(BaseSettings):

    #path
    PATH: str = os.getcwd()

    #model
    TARGET_EXPLANATION: Dict[int, str]
    CATEGORICAL_FEATURES: List[str]
    NUMERICAL_FEATURES: List[str]
    TARGET_PARAMETER: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
