from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(
    title="Neon-FastAPI-Next-Todo API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration - Allow all origins for development/deployment flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for deployment flexibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Neon-FastAPI-Next-Todo API", "version": "1.0.0"}

# Import and include routers
from app.api.routes import tasks, debug, chat
app.include_router(tasks.router)
app.include_router(chat.router)
app.include_router(debug.router)
