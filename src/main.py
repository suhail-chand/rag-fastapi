import os
import getpass
from fastapi import FastAPI
from .doc.router import router 

from .config import settings


if not settings.HF_TOKEN:
    settings.HF_TOKEN = getpass.getpass('HuggingFace Access Token: ')
if not settings.MISTRAL_API_KEY:
    settings.MISTRAL_API_KEY = getpass.getpass('Mistral API Key: ')

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

app.include_router(router, prefix='/doc', tags=['doc'])
