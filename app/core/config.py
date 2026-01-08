from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    BETTER_AUTH_SECRET: str
    FRONTEND_URL: str = "https://todoapp-2-farhan.netlify.app"
    ENVIRONMENT: str = "development"
    API_HOST: str = "https://to-do-app-phase-2-production.up.railway.app"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
