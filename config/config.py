from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()

def get_env_variable(key: str, default: Optional[str] = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set")
    return value

# Critical variables (no default)
#ELASTIC_URL = get_env_variable('ELASTIC_URL')
HUGGING_FACE_API = get_env_variable('HUGGING_FACE_API')
OPENAI_API_KEY = get_env_variable('OPENAI_API_KEY')
PROJ_ID = get_env_variable('PROJ_ID')


#ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
#ELASTIC_CA_CERTS_PATH = os.getenv("ELASTIC_CA_CERTS_PATH")

ELASTIC_ENDPOINT = get_env_variable('ELASTIC_ENDPOINT')
ELASTIC_CLOUD_ID = get_env_variable('ELASTIC_CLOUD_ID')
ELASTIC_API_KEY = get_env_variable('ELASTIC_API_KEY')

SOLR_ENDPOINT = get_env_variable('SOLR_ENDPOINT')

SOLR_PROFILE = get_env_variable('SOLR_PROFILE')