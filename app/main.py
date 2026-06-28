from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.storage.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Production-Ready Codebase RAG", version="1.0.0", lifespan=lifespan)
app.include_router(router)
