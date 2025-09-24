from pydantic import BaseSettings

class Settings(BaseSettings):
    base_model_name: str = "pollen-adaptive-intelligence"
    episodic_memory_capacity: int = 1000
    long_term_memory_path: str = "data/lt_memory.json"
    ethical_guidelines: str = "ethical_guidelines.txt"

    class Config:
        env_file = ".env"

settings = Settings()