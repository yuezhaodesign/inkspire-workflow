from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    perusall_base_url: str = os.getenv("PERUSALL_BASE_URL", "https://api.perusall.com")
    perusall_api_token: str = os.getenv("PERUSALL_API_TOKEN", "")
    perusall_institution: str = os.getenv("PERUSALL_INSTITUTION", "")

settings = Settings()
