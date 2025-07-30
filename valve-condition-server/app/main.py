from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.endpoints import EndpointsManager
from config.settings import settings


endpoints_manager = EndpointsManager()

app = FastAPI()
app.include_router(endpoints_manager.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False
    )
